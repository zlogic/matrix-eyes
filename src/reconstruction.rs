use std::{error, fmt, fs::File, io::BufReader};

use image::{imageops, DynamicImage, ImageDecoder, ImageReader, RgbImage};

use crate::depth_pro;

struct SourceImage {
    img: RgbImage,
    original_size: (u32, u32),
    focal_length_35mm: Option<f32>,
}

impl SourceImage {
    fn load(
        path: &str,
        focal_length_35mm: Option<f32>,
    ) -> Result<SourceImage, ReconstructionError> {
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
        // TODO: use model size
        let img = img
            .resize_exact(1536, 1536, imageops::FilterType::Lanczos3)
            .into_rgb8();

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
        let width = self.img.width() as f64;
        let height = self.img.height() as f64;
        let diagonal = (width * width + height * height).sqrt();
        Some(focal_length_35mm * diagonal / diagonal_35mm)
    }
}

pub struct DepthModel {
    encoder: depth_pro::Encoder,
}

impl DepthModel {
    pub fn new(checkpoint_path: &str) -> Result<DepthModel, ReconstructionError> {
        // TODO: choose the best available device
        let vb = candle_nn::VarBuilder::from_pth(
            checkpoint_path,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;

        let encoder = depth_pro::Encoder::new(vb)?;
        Ok(DepthModel { encoder })
    }

    pub fn extract_depth(
        &self,
        path: &str,
        focal_length_35mm: Option<f32>,
    ) -> Result<(), ReconstructionError> {
        let img = match SourceImage::load(path, focal_length_35mm) {
            Ok(img) => img,
            Err(err) => {
                eprintln!("Failed to load source image: {}", err);
                return Err(err);
            }
        };

        Ok(())
    }
}

#[derive(Debug)]
pub enum ReconstructionError {
    Internal(&'static str),
    Image(image::ImageError),
    Io(std::io::Error),
    Exif(exif::Error),
    Candle(candle_core::Error),
}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Image(ref err) => write!(f, "Image error: {}", err),
            Self::Io(ref err) => write!(f, "IO error: {}", err),
            Self::Exif(ref err) => write!(f, "EXIF error: {}", err),
            Self::Candle(ref err) => write!(f, "Candle error: {}", err),
        }
    }
}

impl std::error::Error for ReconstructionError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Image(ref err) => Some(err),
            Self::Io(ref err) => Some(err),
            Self::Exif(ref err) => Some(err),
            Self::Candle(ref err) => Some(err),
        }
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

impl From<candle_core::Error> for ReconstructionError {
    fn from(e: candle_core::Error) -> ReconstructionError {
        Self::Candle(e)
    }
}

impl From<&'static str> for ReconstructionError {
    fn from(msg: &'static str) -> ReconstructionError {
        Self::Internal(msg)
    }
}
