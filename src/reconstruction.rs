use std::{error, fmt, path::PathBuf};

use burn::{
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder as _},
    tensor::{Float, Shape, Tensor, TensorData},
};
use image::{imageops, DynamicImage, ImageDecoder, ImageReader};

use crate::{depth_pro, output};

#[cfg(any(feature = "ndarray", feature = "ndarray-accelerate"))]
pub type EnabledBackend = burn::backend::NdArray;
#[cfg(any(feature = "wgpu", feature = "wgpu-spirv"))]
pub type EnabledBackend = burn::backend::Wgpu;
#[cfg(feature = "candle-cuda")]
pub type EnabledBackend = burn::backend::Candle;

#[cfg(any(feature = "ndarray", feature = "ndarray-accelerate"))]
pub fn init_device() -> burn::backend::ndarray::NdArrayDevice {
    burn::backend::ndarray::NdArrayDevice::Cpu
}

#[cfg(any(feature = "wgpu", feature = "wgpu-spirv"))]
pub fn init_device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::DefaultDevice
}

#[cfg(feature = "candle-cuda")]
pub fn init_device() -> burn::backend::candle::CandleDevice {
    burn::backend::candle::CandleDevice::cuda(0)
}

struct SourceImage<B>
where
    B: Backend,
{
    img: Tensor<B, 4>,
    original_size: (u32, u32),
    focal_length_35mm: Option<f32>,
}

impl<B> SourceImage<B>
where
    B: Backend,
{
    fn load(
        path: &str,
        focal_length_35mm: Option<f32>,
        device: &B::Device,
    ) -> Result<SourceImage<B>, ReconstructionError> {
        // TODO: use model size
        const WIDTH: usize = 1536;
        const HEIGHT: usize = 1536;
        const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
        const STD: [f32; 3] = [0.5, 0.5, 0.5];
        let mut decoder = ImageReader::open(path)?.into_decoder()?;
        let focal_length_35mm = if let Some(focal_length) = focal_length_35mm {
            Some(focal_length)
        } else if let Some(exif_metadata) = decoder.exif_metadata()? {
            Self::get_focal_length_35mm(exif_metadata)?
        } else {
            None
        };
        let orientation = decoder.orientation()?;
        let mut img = DynamicImage::from_decoder(decoder)?;
        img.apply_orientation(orientation);
        let original_size = (img.width(), img.height());
        let img = img
            .resize_exact(WIDTH as u32, HEIGHT as u32, imageops::FilterType::Lanczos3)
            .into_rgb8();
        let data = img.into_raw();

        let data = TensorData::new(data, Shape::new([HEIGHT, WIDTH, 3]));
        let data = Tensor::<B, 3, Float>::from_data(data.convert::<f32>(), device)
            .permute([2, 0, 1])
            / 255.0;
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([3, 1, 1]);
        let img = ((data - mean) / std).unsqueeze();

        Ok(SourceImage {
            img,
            original_size,
            focal_length_35mm,
        })
    }

    fn get_focal_length_35mm(exif_data: Vec<u8>) -> Result<Option<f32>, exif::Error> {
        let exif = exif::Reader::new().read_raw(exif_data)?;

        if let Some(focal_length) =
            exif.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)
        {
            Ok(focal_length.value.get_uint(0).map(|f| f as f32))
        } else {
            Ok(None)
        }
    }

    fn focal_length_px(&self) -> Option<f64> {
        let focal_length_35mm = self.focal_length_35mm? as f64;
        // Scale focal length: f_img/f_35mm == diagonal / diagonal(24mm x 36mm) (because 35mm equivalent is based on diagonal length)
        let diagonal_35mm = (24.0f64 * 24.0 + 36.0 * 36.0).sqrt();
        let (width, height) = (self.original_size.0 as f64, self.original_size.1 as f64);
        let diagonal = (width * width + height * height).sqrt();
        Some(focal_length_35mm * diagonal / diagonal_35mm)
    }
}

pub fn extract_depth<B>(
    device: &B::Device,
    model: &depth_pro::DepthProModel<B>,
    source_path: &str,
    destination_path: &str,
    focal_length_35mm: Option<f32>,
) -> Result<(), ReconstructionError>
where
    B: Backend,
{
    let img = match SourceImage::<B>::load(source_path, focal_length_35mm, device) {
        Ok(img) => img,
        Err(err) => {
            eprintln!("Failed to load source image: {}", err);
            return Err(err);
        }
    };
    let [_, _, _h, w] = img.img.dims();
    let f_norm = img.focal_length_px().unwrap_or(1.0) / w as f64;

    let inverse_depth = if let Some(inverse_depth) = try_skip_load("depth_map.mpk", device) {
        inverse_depth
    } else {
        let inverse_depth = model.extract_depth(img.img.clone(), f_norm as f32);
        save_tensor("depth_map.mpk", inverse_depth.clone());
        inverse_depth
    };

    let depth_map = match output::DepthMap::new(inverse_depth, img.original_size) {
        Ok(depth_map) => depth_map,
        Err(err) => {
            let err = err.into();
            eprintln!("Failed to read depth data from device: {}", err);
            return Err(err);
        }
    };
    Ok(output::output(depth_map, destination_path)?)
}

fn try_skip_load<B, const D: usize>(filename: &str, device: &B::Device) -> Option<Tensor<B, D>>
where
    B: Backend,
{
    let filename = PathBuf::from(filename);
    if !filename.exists() {
        None
    } else {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .load(filename, device)
            .expect("failed to load tensor")
    }
}

fn save_tensor<B, const D: usize>(filename: &str, tensor: Tensor<B, D>)
where
    B: Backend,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(tensor, filename.into())
        .expect("failed to record tensor");
}

#[derive(Debug)]
pub enum ReconstructionError {
    Internal(&'static str),
    Output(output::OutputError),
    Image(image::ImageError),
    Io(std::io::Error),
    Exif(exif::Error),
    BurnData(burn::tensor::DataError),
}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Output(ref err) => write!(f, "Output error: {}", err),
            Self::Image(ref err) => write!(f, "Image error: {}", err),
            Self::Io(ref err) => write!(f, "IO error: {}", err),
            Self::Exif(ref err) => write!(f, "EXIF error: {}", err),
            Self::BurnData(burn::tensor::DataError::CastError(ref err)) => {
                write!(f, "Burn data cast error: {}", err)
            }
            Self::BurnData(burn::tensor::DataError::TypeMismatch(ref str)) => {
                write!(f, "Burn data type mismatch: {}", str)
            }
        }
    }
}

impl std::error::Error for ReconstructionError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Output(ref err) => Some(err),
            Self::Image(ref err) => Some(err),
            Self::Io(ref err) => Some(err),
            Self::Exif(ref err) => Some(err),
            Self::BurnData(ref _err) => None,
        }
    }
}

impl From<output::OutputError> for ReconstructionError {
    fn from(err: output::OutputError) -> ReconstructionError {
        Self::Output(err)
    }
}

impl From<image::ImageError> for ReconstructionError {
    fn from(e: image::ImageError) -> ReconstructionError {
        Self::Image(e)
    }
}

impl From<std::io::Error> for ReconstructionError {
    fn from(e: std::io::Error) -> ReconstructionError {
        Self::Io(e)
    }
}

impl From<exif::Error> for ReconstructionError {
    fn from(e: exif::Error) -> ReconstructionError {
        Self::Exif(e)
    }
}

impl From<burn::tensor::DataError> for ReconstructionError {
    fn from(e: burn::tensor::DataError) -> ReconstructionError {
        Self::BurnData(e)
    }
}

impl From<&'static str> for ReconstructionError {
    fn from(msg: &'static str) -> ReconstructionError {
        Self::Internal(msg)
    }
}
