use std::{error, fmt, ops::Range, path::Path, sync::Arc};

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::Backend,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder as _, RecorderError},
    tensor::{ElementConversion as _, Tensor, cast::ToElement as _},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use decoder::{MultiresConvDecoder, MultiresConvDecoderConfig};
use encoder::{DepthProEncoder, DepthProEncoderConfig};
use fov::{FOVNetwork, FOVNetworkConfig};

mod decoder;
mod encoder;
mod fov;
mod vit;

type FloatType = crate::reconstruction::FloatType;

#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Option<Conv2d<B>>,
    conv_tr: Option<ConvTranspose2d<B>>,
}

pub const IMG_SIZE: usize = vit::IMG_SIZE * 4;

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

    pub fn extract_depth<B, PL>(
        &self,
        img: Tensor<B, 4>,
        f_norm: Option<FloatType>,
        device: &B::Device,
        pl: Option<PL>,
    ) -> Result<Tensor<B, 2>, ModelError>
    where
        B: Backend,
        PL: ProgressListener,
    {
        const ENCODER_FEATURE_DIMS: [usize; 4] = [256, 512, 1024, 1024];
        const DECODER_FEATURES: usize = 256;

        let pl = SplitProgressListener {
            pl: pl.map(|pl| Arc::new(pl)),
            range: 0.0..1.0,
        };
        let (pl, pl_fov) = if f_norm.is_none() {
            pl.split_range(0.8)
        } else {
            pl.split_range(1.0)
        };
        let (pl, next_pl) = pl.split_range(0.8);

        let encodings = {
            let (pl, pl_encoder) = pl.split_range(0.05);
            let encoder =
                DepthProEncoderConfig::init(&ENCODER_FEATURE_DIMS, DECODER_FEATURES, device);
            let encoder = PartEncoder { encoder };
            pl.update_message("loading encoder model".into());
            let encoder = self
                .load_record(encoder, "encoder", device)
                .map_err(|err| ModelError::Internal("Failed to load depth model", err))?
                .encoder;
            pl.report_status(1.0);
            encoder.forward_encodings(img.clone(), pl_encoder)
        };
        let (pl, next_pl) = next_pl.split_range(0.98);

        let (features, features_0) = {
            let (pl, pl_decoder) = pl.split_range(0.05);
            let mut dims_encoder = vec![DECODER_FEATURES];
            dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
            let decoder = MultiresConvDecoderConfig::init(&dims_encoder, DECODER_FEATURES, device);
            let decoder = PartDecoder { decoder };
            pl.update_message("loading decoder model".into());
            let decoder = self
                .load_record(decoder, "decoder", device)
                .map_err(|err| ModelError::Internal("Failed to load decoder model", err))?
                .decoder;
            pl.report_status(1.0);
            decoder.forward(encodings, pl_decoder)
        };
        let pl = next_pl;

        let canonical_inverse_depth = {
            let head = HeadConfig {
                dim_decoder: DECODER_FEATURES,
                last_dims: [32, 1],
            }
            .init(device);
            let head = PartHead { head };
            pl.update_message("loading head".into());
            let head = self
                .load_record(head, "head", device)
                .map_err(|err| ModelError::Internal("Failed to load head model", err))?
                .head;
            pl.report_status(0.05);

            pl.update_message("forwarding head".into());

            let features = head[0].forward(features);
            pl.report_status(0.3);
            let features = head[1].forward(features);
            pl.report_status(0.6);
            let features = head[2].forward(features);
            pl.report_status(0.8);
            let features = Relu::new().forward(features);
            pl.report_status(0.9);
            let features = head[3].forward(features);
            pl.report_status(0.95);
            Relu::new().forward(features)
        };

        let canonical_inverse_depth = canonical_inverse_depth.squeeze::<3>(0).squeeze::<2>(0);

        let f_norm = if let Some(f_norm) = f_norm {
            f_norm
        } else {
            let fov = FOVNetworkConfig::init(DECODER_FEATURES, device);
            let fov = PartFOV { fov };
            let (pl, pl_fov) = pl_fov.split_range(0.05);
            pl.update_message("loading fov".into());
            let fov = self
                .load_record(fov, "fov", device)
                .map_err(|err| ModelError::Internal("Failed to load fov model", err))?
                .fov;
            pl.report_status(1.0);

            let fov_deg = fov
                .forward(img, features_0, pl_fov)
                .into_scalar()
                .elem::<B::FloatElem>()
                .to_f32();
            FloatType::from_f32((0.5 * (fov_deg * std::f32::consts::PI / 180.0)).tan() / 0.5)
        };

        let inverse_depth = canonical_inverse_depth.div_scalar(f_norm);
        Ok(inverse_depth.clamp(FloatType::from_f32(1e-4), FloatType::from_f32(1e4)))
    }
}

pub trait ProgressListener
where
    Self: Send + Sync + Sized,
{
    fn report_status(&self, pos: f32);
    fn update_message(&self, status_message: String);
}

struct SplitProgressListener<PL: ProgressListener> {
    pl: Option<Arc<PL>>,
    range: Range<f32>,
}

impl<PL> SplitProgressListener<PL>
where
    PL: ProgressListener,
{
    fn split_range(
        self,
        split_position: f32,
    ) -> (SplitProgressListener<PL>, SplitProgressListener<PL>) {
        let mid = self.range.start + (self.range.end - self.range.start) * split_position;
        let range_left = self.range.start..mid;
        let range_right = mid..self.range.end;
        (
            SplitProgressListener {
                pl: self.pl.clone(),
                range: range_left,
            },
            SplitProgressListener {
                pl: self.pl.clone(),
                range: range_right,
            },
        )
    }
}

impl<PL> ProgressListener for SplitProgressListener<PL>
where
    PL: ProgressListener,
{
    fn report_status(&self, pos: f32) {
        if let Some(pl) = self.pl.as_deref() {
            pl.report_status(self.range.start + pos * (self.range.end - self.range.start));
        }
    }

    fn update_message(&self, status_message: String) {
        if let Some(pl) = self.pl.as_deref() {
            pl.update_message(status_message);
        }
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
