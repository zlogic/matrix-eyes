use std::{error, fmt, ops::Range, path::Path, rc::Rc, sync::Arc};

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::Backend,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder as _},
    store::{
        Applier, KeyRemapper, ModuleAdapter, ModuleStore, PyTorchToBurnAdapter, TensorSnapshot,
        pytorch::PytorchStore,
    },
    tensor::{DType, ElementConversion as _, Tensor, cast::ToElement as _},
};
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
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
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
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
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

#[derive(Clone)]
struct HalfPrecisionAdapter {
    target_dtype: DType,
}

impl HalfPrecisionAdapter {
    fn new(target_dtype: DType) -> Self {
        HalfPrecisionAdapter { target_dtype }
    }
}

impl ModuleAdapter for HalfPrecisionAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // This is a custom version of Burn's HalfPrecisionAdapter.
        // It targets the actual DType instead of swapping between f32 and f16.
        let target_dtype = self.target_dtype;
        if target_dtype == snapshot.dtype {
            return snapshot.clone();
        }
        let original_data_fn = snapshot.clone_data_fn();

        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(target_dtype))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            self.target_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
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
        dtype: DType,
    ) -> Result<M, LoaderError>
    where
        M: Module<B>,
        B: Backend,
    {
        let pytorch_remapper = KeyRemapper::new()
            // Label upsampling blocks to guide enum deserialization.
            .add_pattern(
                "^(encoder\\.upsample[^.]+)\\.0\\.weight",
                "$1.0.conv.weight",
            )
            .map_err(|_| LoaderError::Regex)?
            .add_pattern(
                "^(encoder\\.upsample[^.]+)\\.([0-9]+)\\.weight",
                "$1.$2.conv_tr.weight",
            )
            .map_err(|_| LoaderError::Regex)?
            // Label head blocks to guide enum deserialization.
            .add_pattern("^head\\.0\\.(.+)", "head.0.conv.$1")
            .map_err(|_| LoaderError::Regex)?
            .add_pattern("^head\\.1\\.(.+)", "head.1.conv_tr.$1")
            .map_err(|_| LoaderError::Regex)?
            .add_pattern("^head\\.2\\.(.+)", "head.2.conv.$1")
            .map_err(|_| LoaderError::Regex)?
            .add_pattern("^head\\.4\\.(.+)", "head.4.conv.$1")
            .map_err(|_| LoaderError::Regex)?
            // Label fov encoder to avoid using vec/enums.
            .add_pattern("^fov.encoder\\.0\\.(.+)", "fov.encoder.fov_encoder.$1")
            .map_err(|_| LoaderError::Regex)?
            .add_pattern("^fov.encoder\\.1\\.(.+)", "fov.encoder.linear.$1")
            .map_err(|_| LoaderError::Regex)?;
        let converted_filename = Path::new(&self.checkpoint_path);
        let converted_filename = converted_filename
            .with_file_name(
                converted_filename
                    .file_stem()
                    .and_then(|filename| filename.to_str())
                    .map_or(suffix.to_string(), |filename| {
                        format!("{filename}-{suffix}")
                    }),
            )
            .with_extension("mpk")
            .to_path_buf();

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
        if converted_filename.exists() {
            let record = recorder.load(converted_filename, device)?;
            Ok(model.load_record(record))
        } else {
            let mut store =
                PytorchStore::from_file(self.checkpoint_path.as_str()).remap(pytorch_remapper);
            // This is messy because PytorchStore doesn't have an easy way to convert f16 into f32.
            let snapshots: Vec<TensorSnapshot> =
                store.get_all_snapshots()?.values().cloned().collect();
            let adapter = PyTorchToBurnAdapter.chain(HalfPrecisionAdapter::new(dtype));
            let mut applier = Applier::new(snapshots, None, Some(Box::new(adapter)), true);
            let model = model.map(&mut applier);
            let result = applier.into_result();
            if !result.errors.is_empty() {
                return Err(LoaderError::RecorderErrors(result.errors));
            }
            if !result.missing.is_empty() {
                return Err(LoaderError::RecorderMissing(result.missing));
            }
            if self.convert_checkpoints {
                recorder.record(model.clone().into_record(), converted_filename)?;
            }
            Ok(model)
        }
    }

    pub fn extract_depth<B, PL>(
        &self,
        img: Tensor<B, 4>,
        f_norm: Option<f32>,
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
                .load_record(encoder, "encoder", device, img.dtype())
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
                .load_record(decoder, "decoder", device, img.dtype())
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
                .load_record(head, "head", device, img.dtype())
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

        let canonical_inverse_depth = canonical_inverse_depth
            .squeeze_dim::<3>(0)
            .squeeze_dim::<2>(0);

        let f_norm = if let Some(f_norm) = f_norm {
            f_norm
        } else {
            let fov = FOVNetworkConfig::init(DECODER_FEATURES, device);
            let fov = PartFOV { fov };
            let (pl, pl_fov) = pl_fov.split_range(0.05);
            pl.update_message("loading fov".into());
            let fov = self
                .load_record(fov, "fov", device, img.dtype())
                .map_err(|err| ModelError::Internal("Failed to load fov model", err))?
                .fov;
            pl.report_status(1.0);

            let fov_deg = fov
                .forward(img, features_0, pl_fov)
                .into_scalar()
                .elem::<B::FloatElem>()
                .to_f32();
            (0.5 * (fov_deg * std::f32::consts::PI / 180.0)).tan() / 0.5
        };

        let inverse_depth = canonical_inverse_depth.div_scalar(f_norm);
        Ok(inverse_depth.clamp(1e-4, 1e4))
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
pub enum LoaderError {
    Regex,
    Recorder(burn::record::RecorderError),
    RecorderErrors(Vec<burn::store::ApplyError>),
    RecorderMissing(Vec<(String, String)>),
    Pytorch(burn::store::PytorchStoreError),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Regex => write!(f, "Regex error"),
            Self::Recorder(ref err) => write!(f, "Recorder error: {err}"),
            Self::RecorderErrors(ref errs) => {
                write!(f, "Recorder errors: ")?;
                for (i, err) in errs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", {err}")?;
                    } else {
                        write!(f, "{err}")?;
                    }
                }
                Ok(())
            }
            Self::RecorderMissing(ref items) => {
                write!(f, "Recorder missing items: ")?;
                for (i, (path, stack)) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", {path}={stack}")?;
                    } else {
                        write!(f, "{path}={stack}")?;
                    }
                }
                Ok(())
            }
            Self::Pytorch(ref err) => write!(f, "PyTorch store error: {err}"),
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Regex => None,
            Self::Recorder(ref err) => Some(err),
            Self::RecorderErrors(ref _errs) => None,
            Self::RecorderMissing(ref _items) => None,
            Self::Pytorch(ref err) => Some(err),
        }
    }
}

impl From<burn::record::RecorderError> for LoaderError {
    fn from(e: burn::record::RecorderError) -> LoaderError {
        Self::Recorder(e)
    }
}

impl From<burn::store::PytorchStoreError> for LoaderError {
    fn from(e: burn::store::PytorchStoreError) -> LoaderError {
        Self::Pytorch(e)
    }
}

#[derive(Debug)]
pub enum ModelError {
    Internal(&'static str, LoaderError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg, ref err) => write!(f, "Model error: {msg}: {err}"),
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
