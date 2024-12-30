use std::{error, fmt, fs::File, io::BufReader};

use image::RgbImage;

struct SourceImage {
    img: RgbImage,
    focal_length_35mm: Option<f32>,
}

impl SourceImage {
    fn load(
        path: &str,
        focal_length_35mm: Option<f32>,
    ) -> Result<SourceImage, ReconstructionError> {
        let focal_length_35mm = if let Some(focal_length) = focal_length_35mm {
            Some(focal_length)
        } else {
            Self::get_focal_length_35mm(path)?
        };
        let img = image::open(path)?.into_rgb8();

        Ok(SourceImage {
            img,
            focal_length_35mm,
        })
    }

    fn get_focal_length_35mm(path: &str) -> Result<Option<f32>, exif::Error> {
        let mut reader = BufReader::new(File::open(path)?);
        let exif = exif::Reader::new().read_from_container(&mut reader)?;

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

pub fn extract_depth(
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

#[derive(Debug)]
pub enum ReconstructionError {
    Internal(&'static str),
    Image(image::ImageError),
    Exif(exif::Error),
}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Image(ref err) => write!(f, "Image error: {}", err),
            Self::Exif(ref err) => write!(f, "EXIF error: {}", err),
        }
    }
}

impl std::error::Error for ReconstructionError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Image(ref err) => Some(err),
            Self::Exif(ref err) => Some(err),
        }
    }
}

impl From<image::ImageError> for ReconstructionError {
    fn from(e: image::ImageError) -> ReconstructionError {
        Self::Image(e)
    }
}

impl From<exif::Error> for ReconstructionError {
    fn from(e: exif::Error) -> ReconstructionError {
        Self::Exif(e)
    }
}

impl From<&'static str> for ReconstructionError {
    fn from(msg: &'static str) -> ReconstructionError {
        Self::Internal(msg)
    }
}
