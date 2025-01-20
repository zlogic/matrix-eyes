use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d, Relu,
    },
    prelude::Backend,
    record::{FullPrecisionSettings, Recorder as _, RecorderError},
    tensor::Tensor,
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use decoder::{MultiresConvDecoder, MultiresConvDecoderConfig};
use encoder::{DepthProEncoder, DepthProEncoderConfig};

mod decoder;
mod encoder;

#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Option<Conv2d<B>>,
    conv_tr: Option<ConvTranspose2d<B>>,
}

impl<B> ConvBlock<B>
where
    B: Backend,
{
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match (&self.conv, &self.conv_tr) {
            (Some(conv), None) => conv.forward(input),
            (None, Some(conv_tr)) => conv_tr.forward(input),
            (None, None) => panic!("block is empty"),
            (Some(_), Some(_)) => {
                panic!("block has convolution and transposed convolution at the same time")
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct DepthProModel<B: Backend> {
    encoder: DepthProEncoder<B>,
    decoder: MultiresConvDecoder<B>,
    head: Vec<ConvBlock<B>>,
}

#[derive(Config, Debug)]
struct HeadConfig {
    dim_decoder: usize,
    last_dims: [usize; 2],
}

impl HeadConfig {
    fn init<B>(&self, device: &B::Device) -> Vec<ConvBlock<B>>
    where
        B: Backend,
    {
        vec![
            ConvBlock {
                conv: Some(
                    Conv2dConfig::new([self.dim_decoder, self.dim_decoder / 2], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1))
                        .init(device),
                ),
                conv_tr: None,
            },
            ConvBlock {
                conv: None,
                conv_tr: Some(
                    ConvTranspose2dConfig::new(
                        [self.dim_decoder / 2, self.dim_decoder / 2],
                        [2, 2],
                    )
                    .with_stride([2, 2])
                    .init(device),
                ),
            },
            ConvBlock {
                conv: Some(
                    Conv2dConfig::new([self.dim_decoder / 2, self.last_dims[0]], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1))
                        .init(device),
                ),
                conv_tr: None,
            },
            ConvBlock {
                conv: Some(
                    Conv2dConfig::new([self.last_dims[0], self.last_dims[1]], [1, 1]).init(device),
                ),
                conv_tr: None,
            },
        ]
    }
}

impl<B> DepthProModel<B>
where
    B: Backend,
{
    pub fn new(checkpoint_path: &str, device: &B::Device) -> Result<DepthProModel<B>, RecorderError>
    where
        B: Backend,
    {
        const ENCODER_FEATURE_DIMS: [usize; 4] = [256, 512, 1024, 1024];
        const DECODER_FEATURES: usize = 256;

        let load_args = LoadArgs::new(checkpoint_path.into())
            // Label upsampling blocks to guide enum deserialization.
            .with_key_remap("(encoder.upsample[^.]+)\\.0\\.weight", "$1.0.conv.weight")
            .with_key_remap(
                "(encoder.upsample[^.]+)\\.([0-9]+)\\.weight",
                "$1.$2.conv_tr.weight",
            )
            // Label head blocks to guide enum deserialization.
            .with_key_remap("head\\.0\\.(.+)", "head.0.conv.$1")
            .with_key_remap("head\\.1\\.(.+)", "head.1.conv_tr.$1")
            .with_key_remap("head\\.2\\.(.+)", "head.2.conv.$1")
            .with_key_remap("head\\.4\\.(.+)", "head.4.conv.$1");
        let record: DepthProModelRecord<B> =
            PyTorchFileRecorder::<FullPrecisionSettings>::default().load(load_args, device)?;

        let encoder = DepthProEncoderConfig::init(&ENCODER_FEATURE_DIMS, DECODER_FEATURES, device);

        let mut dims_encoder = vec![DECODER_FEATURES];
        dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
        let decoder = MultiresConvDecoderConfig::init(&dims_encoder, DECODER_FEATURES, device);
        let head = HeadConfig {
            dim_decoder: DECODER_FEATURES,
            last_dims: [32, 1],
        }
        .init(device);

        let model = DepthProModel::<B> {
            encoder,
            decoder,
            head,
        };
        Ok(model.load_record(record))
    }

    pub fn extract_depth(&self, img: Tensor<B, 4>, f_norm: f32) -> Tensor<B, 2>
    where
        B: Backend,
    {
        let encodings = self.encoder.forward_encodings(img);

        let (features, features_0) = self.decoder.forward(encodings);
        drop(features_0);

        let features = self.head[0].forward(features);
        let features = self.head[1].forward(features);
        let features = self.head[2].forward(features);
        let features = Relu::new().forward(features);
        let features = self.head[3].forward(features);
        let canonical_inverse_depth = Relu::new().forward(features);

        let canonical_inverse_depth = canonical_inverse_depth.squeeze::<3>(0).squeeze::<2>(0);

        let inverse_depth = canonical_inverse_depth.div_scalar(f_norm);
        inverse_depth.clamp(1e-4, 1e4)
    }
}
