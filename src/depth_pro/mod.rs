use std::{error, fmt, path::Path};

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d, Relu,
    },
    prelude::Backend,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder as _, RecorderError},
    tensor::{cast::ToElement as _, Tensor},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use decoder::{MultiresConvDecoder, MultiresConvDecoderConfig};
use encoder::{DepthProEncoder, DepthProEncoderConfig};
use fov::{FOVNetwork, FOVNetworkConfig};

mod decoder;
mod encoder;
mod fov;
mod vit;

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

#[derive(Module, Debug)]
struct PartEncoder<B: Backend> {
    encoder: DepthProEncoder<B>,
}

#[derive(Module, Debug)]
struct PartDecoder<B: Backend> {
    decoder: MultiresConvDecoder<B>,
}

#[derive(Module, Debug)]
struct PartHead<B: Backend> {
    head: Vec<ConvBlock<B>>,
}

#[derive(Module, Debug)]
struct PartFOV<B: Backend> {
    fov: FOVNetwork<B>,
}

pub struct DepthProModelLoader {
    checkpoint_path: String,
    convert_checkpoints: bool,
}

impl DepthProModelLoader {
    pub fn new(checkpoint_path: &str, convert_checkpoints: bool) -> DepthProModelLoader {
        DepthProModelLoader {
            checkpoint_path: checkpoint_path.to_string(),
            convert_checkpoints,
        }
    }

    fn load_record<M, B>(
        &self,
        model: M,
        suffix: &str,
        device: &B::Device,
    ) -> Result<M, RecorderError>
    where
        M: Module<B>,
        B: Backend,
    {
        let pytorch_load_args = LoadArgs::new(self.checkpoint_path.as_str().into())
            // Label upsampling blocks to guide enum deserialization.
            .with_key_remap(
                "^(encoder\\.upsample[^.]+)\\.0\\.weight",
                "$1.0.conv.weight",
            )
            .with_key_remap(
                "^(encoder\\.upsample[^.]+)\\.([0-9]+)\\.weight",
                "$1.$2.conv_tr.weight",
            )
            // Label head blocks to guide enum deserialization.
            .with_key_remap("^head\\.0\\.(.+)", "head.0.conv.$1")
            .with_key_remap("^head\\.1\\.(.+)", "head.1.conv_tr.$1")
            .with_key_remap("^head\\.2\\.(.+)", "head.2.conv.$1")
            .with_key_remap("^head\\.4\\.(.+)", "head.4.conv.$1")
            // Label fov encoder to avoid using vec/enums.
            .with_key_remap("^fov.encoder\\.0\\.(.+)", "fov.encoder.fov_encoder.$1")
            .with_key_remap("^fov.encoder\\.1\\.(.+)", "fov.encoder.linear.$1");

        let converted_filename = Path::new(&self.checkpoint_path);
        let converted_filename = converted_filename
            .with_file_name(
                converted_filename
                    .file_stem()
                    .and_then(|filename| filename.to_str())
                    .map_or(suffix.to_string(), |filename| {
                        format!("{}-{}", filename, suffix)
                    }),
            )
            .with_extension("mpk")
            .to_path_buf();

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
        let record: M::Record = if converted_filename.exists() {
            recorder.load(converted_filename, device)?
        } else {
            let record = PyTorchFileRecorder::<HalfPrecisionSettings>::default()
                .load(pytorch_load_args.clone(), device)?;
            if self.convert_checkpoints {
                recorder.record(record, converted_filename.clone())?;
                recorder.load(converted_filename, device)?
            } else {
                record
            }
        };
        Ok(model.load_record(record))
    }

    pub fn extract_depth<B>(
        &self,
        img: Tensor<B, 4>,
        f_norm: Option<f32>,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, ModelError>
    where
        B: Backend,
    {
        const ENCODER_FEATURE_DIMS: [usize; 4] = [256, 512, 1024, 1024];
        const DECODER_FEATURES: usize = 256;

        let encodings = {
            let encoder =
                DepthProEncoderConfig::init(&ENCODER_FEATURE_DIMS, DECODER_FEATURES, device);
            let encoder = PartEncoder { encoder };
            let encoder = self
                .load_record(encoder, "encoder", device)
                .map_err(|err| ModelError::Internal("Failed to load depth model", err))?
                .encoder;
            encoder.forward_encodings(img.clone())
        };

        let (features, features_0) = {
            let mut dims_encoder = vec![DECODER_FEATURES];
            dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
            let decoder = MultiresConvDecoderConfig::init(&dims_encoder, DECODER_FEATURES, device);
            let decoder = PartDecoder { decoder };
            let decoder = self
                .load_record(decoder, "decoder", device)
                .map_err(|err| ModelError::Internal("Failed to load decoder model", err))?
                .decoder;
            decoder.forward(encodings)
        };

        let canonical_inverse_depth = {
            let head = HeadConfig {
                dim_decoder: DECODER_FEATURES,
                last_dims: [32, 1],
            }
            .init(device);
            let head = PartHead { head };
            let head = self
                .load_record(head, "head", device)
                .map_err(|err| ModelError::Internal("Failed to load head model", err))?
                .head;

            let features = head[0].forward(features);
            let features = head[1].forward(features);
            let features = head[2].forward(features);
            let features = Relu::new().forward(features);
            let features = head[3].forward(features);
            Relu::new().forward(features)
        };

        let canonical_inverse_depth = canonical_inverse_depth.squeeze::<3>(0).squeeze::<2>(0);

        let f_norm = if let Some(f_norm) = f_norm {
            f_norm
        } else {
            let fov = FOVNetworkConfig::init(DECODER_FEATURES, device);
            let fov = PartFOV { fov };
            let fov = self
                .load_record(fov, "fov", device)
                .map_err(|err| ModelError::Internal("Failed to load fov model", err))?
                .fov;

            let fov_deg = fov.forward(img, features_0).into_scalar().to_f32();
            (0.5 * (fov_deg * std::f32::consts::PI / 180.0)).tan() / 0.5
        };

        let inverse_depth = canonical_inverse_depth.div_scalar(f_norm);
        Ok(inverse_depth.clamp(1e-4, 1e4))
    }
}

#[derive(Debug)]
pub enum ModelError {
    Internal(&'static str, RecorderError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg, ref err) => write!(f, "Model error: {}: {}", msg, err),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg, ref err) => Some(err),
        }
    }
}
