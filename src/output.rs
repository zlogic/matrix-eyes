use std::{fmt, ops::Range};

use burn::{
    prelude::Backend,
    tensor::{DataError, Tensor},
};
use image::{imageops, DynamicImage, Rgb, RgbImage};
use rand::Rng as _;

pub struct DepthMap {
    data: Vec<f32>,
    data_width: usize,
    data_height: usize,
    original_width: u32,
    original_height: u32,
}

#[derive(Debug, Clone)]
pub enum ImageOutputFormat {
    DepthMap,
    Stereogram(Option<f32>, f32),
}

impl DepthMap {
    pub fn new<B>(
        inverse_depth: Tensor<B, 2>,
        original_size: (u32, u32),
    ) -> Result<DepthMap, DataError>
    where
        B: Backend,
    {
        let [data_width, data_height] = inverse_depth.dims();
        let data = inverse_depth.to_data().to_vec()?;
        let (original_width, original_height) = original_size;

        Ok(DepthMap {
            data,
            data_width,
            data_height,
            original_width,
            original_height,
        })
    }

    fn inverse_depth_range(&self) -> Range<f32> {
        self.data
            .iter()
            .fold(self.data[0]..self.data[0], |acc, val| {
                acc.start.min(*val)..acc.end.max(*val)
            })
    }

    #[inline]
    fn depth_value(&self, x: usize, y: usize) -> f32 {
        self.data[self.data_height * y + x]
    }

    #[inline]
    fn interpolate_point(&self, x: f32, y: f32) -> f32 {
        let x = (x * self.data_width as f32).max(0.0);
        let y = (y * self.data_height as f32).max(0.0);
        let x0 = (x.floor() as usize).clamp(0, self.data_width - 1);
        let y0 = (y.floor() as usize).clamp(0, self.data_height - 1);
        let x1 = (x0 + 1).clamp(0, self.data_width - 1);
        let y1 = (y0 + 1).clamp(0, self.data_height - 1);
        let x = x.fract();
        let y = y.fract();

        // Bilinear interpolation.
        (1.0 - x) * (1.0 - y) * self.depth_value(x0, y0)
            + x * (1.0 - y) * self.depth_value(x1, y0)
            + (1.0 - x) * y * self.depth_value(x0, y1)
            + x * y * self.depth_value(x1, y1)
    }

    pub fn output_image(
        &self,
        destination_path: &str,
        format: ImageOutputFormat,
    ) -> Result<(), OutputError> {
        match format {
            ImageOutputFormat::DepthMap => self.output_depth_map(destination_path),
            ImageOutputFormat::Stereogram(resize_scale, amplitude) => {
                self.output_stereogram(destination_path, resize_scale, amplitude)
            }
        }
    }

    fn output_depth_map(&self, destination_path: &str) -> Result<(), OutputError> {
        let mut out_image = RgbImage::new(self.data_width as u32, self.data_height as u32);

        let depth_range = self.inverse_depth_range();
        let (min_depth, max_depth) = (depth_range.start, depth_range.end);
        for ((_x, _y, pixel), depth) in out_image.enumerate_pixels_mut().zip(self.data.iter()) {
            let depth = (max_depth - depth) / (max_depth - min_depth);
            *pixel = map_depth(depth);
        }

        let out_image = DynamicImage::from(out_image).resize_exact(
            self.original_width,
            self.original_height,
            imageops::Lanczos3,
        );
        Ok(out_image.save(destination_path)?)
    }

    fn output_stereogram(
        &self,
        destination_path: &str,
        resize_scale: Option<f32>,
        amplitude: f32,
    ) -> Result<(), OutputError> {
        let (output_width, output_height) = if let Some(resize_scale) = resize_scale {
            (
                ((self.original_width as f32) * resize_scale).round() as u32,
                ((self.original_height as f32) * resize_scale).round() as u32,
            )
        } else {
            (self.original_width, self.original_height)
        };
        let mut out_image = RgbImage::new(output_width, output_height);

        let depth_range = self.inverse_depth_range();
        let (min_depth, max_depth) = (depth_range.start, depth_range.end);

        let depth_multiplier = output_width as f32 * amplitude;
        let pattern_width = (depth_multiplier * 2.0).round() as usize + 16;

        let mut rng = rand::thread_rng();
        for (y, row) in out_image.enumerate_rows_mut() {
            let noise_row = (0..output_width)
                .map(|_x| {
                    let mut rgb = Rgb::from([0u8, 0u8, 0u8]);
                    rng.fill(&mut rgb.0);
                    rgb
                })
                .collect::<Vec<_>>();
            let mut output_row = noise_row.clone();
            for x in 0..output_width {
                let depth = self.interpolate_point(
                    x as f32 / output_width as f32,
                    y as f32 / output_height as f32,
                );
                let depth = (depth - min_depth) / (max_depth - min_depth);
                let x = x as usize;
                output_row[x] = if x >= pattern_width {
                    let shift = (depth * depth_multiplier).round() as usize;
                    output_row[x + shift - pattern_width]
                } else {
                    noise_row[x % pattern_width]
                };
            }
            for ((_x, _y, pixel), noise_value) in row.zip(output_row) {
                *pixel = noise_value
            }
        }

        Ok(out_image.save(destination_path)?)
    }
}

#[inline]
fn map_depth(value: f32) -> Rgb<u8> {
    // viridis from https://bids.github.io/colormap/
    const COLORMAP_R: [u8; 256] = [
        0xfd, 0xfb, 0xf8, 0xf6, 0xf4, 0xf1, 0xef, 0xec, 0xea, 0xe7, 0xe5, 0xe2, 0xdf, 0xdd, 0xda,
        0xd8, 0xd5, 0xd2, 0xd0, 0xcd, 0xca, 0xc8, 0xc5, 0xc2, 0xc0, 0xbd, 0xba, 0xb8, 0xb5, 0xb2,
        0xb0, 0xad, 0xaa, 0xa8, 0xa5, 0xa2, 0xa0, 0x9d, 0x9b, 0x98, 0x95, 0x93, 0x90, 0x8e, 0x8b,
        0x89, 0x86, 0x84, 0x81, 0x7f, 0x7c, 0x7a, 0x77, 0x75, 0x73, 0x70, 0x6e, 0x6c, 0x69, 0x67,
        0x65, 0x63, 0x60, 0x5e, 0x5c, 0x5a, 0x58, 0x56, 0x54, 0x52, 0x50, 0x4e, 0x4c, 0x4a, 0x48,
        0x46, 0x44, 0x42, 0x40, 0x3f, 0x3d, 0x3b, 0x3a, 0x38, 0x37, 0x35, 0x34, 0x32, 0x31, 0x2f,
        0x2e, 0x2d, 0x2c, 0x2a, 0x29, 0x28, 0x27, 0x26, 0x25, 0x25, 0x24, 0x23, 0x22, 0x22, 0x21,
        0x21, 0x20, 0x20, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1e, 0x1e, 0x1e, 0x1f, 0x1f, 0x1f,
        0x1f, 0x1f, 0x1f, 0x1f, 0x20, 0x20, 0x20, 0x21, 0x21, 0x21, 0x21, 0x22, 0x22, 0x22, 0x23,
        0x23, 0x23, 0x24, 0x24, 0x25, 0x25, 0x25, 0x26, 0x26, 0x26, 0x27, 0x27, 0x27, 0x28, 0x28,
        0x29, 0x29, 0x29, 0x2a, 0x2a, 0x2a, 0x2b, 0x2b, 0x2c, 0x2c, 0x2c, 0x2d, 0x2d, 0x2e, 0x2e,
        0x2e, 0x2f, 0x2f, 0x30, 0x30, 0x31, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35,
        0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x3a, 0x3a, 0x3b, 0x3b, 0x3c, 0x3c,
        0x3d, 0x3d, 0x3e, 0x3e, 0x3e, 0x3f, 0x3f, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x42, 0x43,
        0x43, 0x44, 0x44, 0x44, 0x45, 0x45, 0x45, 0x46, 0x46, 0x46, 0x46, 0x47, 0x47, 0x47, 0x47,
        0x47, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48,
        0x48, 0x48, 0x48, 0x47, 0x47, 0x47, 0x47, 0x47, 0x46, 0x46, 0x46, 0x46, 0x45, 0x45, 0x44,
        0x44,
    ];
    const COLORMAP_G: [u8; 256] = [
        0xe7, 0xe7, 0xe6, 0xe6, 0xe6, 0xe5, 0xe5, 0xe5, 0xe5, 0xe4, 0xe4, 0xe4, 0xe3, 0xe3, 0xe3,
        0xe2, 0xe2, 0xe2, 0xe1, 0xe1, 0xe1, 0xe0, 0xe0, 0xdf, 0xdf, 0xdf, 0xde, 0xde, 0xde, 0xdd,
        0xdd, 0xdc, 0xdc, 0xdb, 0xdb, 0xda, 0xda, 0xd9, 0xd9, 0xd8, 0xd8, 0xd7, 0xd7, 0xd6, 0xd6,
        0xd5, 0xd5, 0xd4, 0xd3, 0xd3, 0xd2, 0xd1, 0xd1, 0xd0, 0xd0, 0xcf, 0xce, 0xcd, 0xcd, 0xcc,
        0xcb, 0xcb, 0xca, 0xc9, 0xc8, 0xc8, 0xc7, 0xc6, 0xc5, 0xc5, 0xc4, 0xc3, 0xc2, 0xc1, 0xc1,
        0xc0, 0xbf, 0xbe, 0xbd, 0xbc, 0xbc, 0xbb, 0xba, 0xb9, 0xb8, 0xb7, 0xb6, 0xb6, 0xb5, 0xb4,
        0xb3, 0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xad, 0xac, 0xab, 0xaa, 0xa9, 0xa8, 0xa7, 0xa6,
        0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa1, 0xa0, 0x9f, 0x9e, 0x9d, 0x9c, 0x9b, 0x9a, 0x99, 0x98,
        0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x92, 0x91, 0x90, 0x8f, 0x8e, 0x8d, 0x8c, 0x8b, 0x8a,
        0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x82, 0x81, 0x80, 0x7f, 0x7e, 0x7d, 0x7c,
        0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71, 0x71, 0x70, 0x6f, 0x6e,
        0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60, 0x5f,
        0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50, 0x4f,
        0x4e, 0x4d, 0x4c, 0x4a, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x42, 0x41, 0x40, 0x3f, 0x3e,
        0x3d, 0x3b, 0x3a, 0x39, 0x38, 0x37, 0x35, 0x34, 0x33, 0x32, 0x30, 0x2f, 0x2e, 0x2d, 0x2c,
        0x2a, 0x29, 0x28, 0x26, 0x25, 0x24, 0x23, 0x21, 0x20, 0x1f, 0x1d, 0x1c, 0x1b, 0x1a, 0x18,
        0x17, 0x16, 0x14, 0x13, 0x11, 0x10, 0x0e, 0x0d, 0x0b, 0x0a, 0x08, 0x07, 0x05, 0x04, 0x02,
        0x01,
    ];
    const COLORMAP_B: [u8; 256] = [
        0x25, 0x23, 0x21, 0x20, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x19, 0x18, 0x18, 0x18, 0x19,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x23, 0x25, 0x26, 0x28, 0x29, 0x2b, 0x2d,
        0x2f, 0x30, 0x32, 0x34, 0x36, 0x37, 0x39, 0x3b, 0x3c, 0x3e, 0x40, 0x41, 0x43, 0x45, 0x46,
        0x48, 0x49, 0x4b, 0x4d, 0x4e, 0x50, 0x51, 0x53, 0x54, 0x56, 0x57, 0x58, 0x5a, 0x5b, 0x5c,
        0x5e, 0x5f, 0x60, 0x62, 0x63, 0x64, 0x65, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e,
        0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x79, 0x7a, 0x7b, 0x7c,
        0x7c, 0x7d, 0x7e, 0x7f, 0x7f, 0x80, 0x81, 0x81, 0x82, 0x82, 0x83, 0x83, 0x84, 0x85, 0x85,
        0x85, 0x86, 0x86, 0x87, 0x87, 0x88, 0x88, 0x88, 0x89, 0x89, 0x89, 0x8a, 0x8a, 0x8a, 0x8b,
        0x8b, 0x8b, 0x8b, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d,
        0x8d, 0x8d, 0x8d, 0x8d, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8b, 0x8b, 0x8b, 0x8b, 0x8a,
        0x8a, 0x8a, 0x8a, 0x89, 0x89, 0x89, 0x88, 0x88, 0x88, 0x87, 0x87, 0x86, 0x86, 0x85, 0x85,
        0x84, 0x84, 0x83, 0x83, 0x82, 0x81, 0x81, 0x80, 0x7f, 0x7e, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a,
        0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x71, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6a,
        0x69, 0x68, 0x67, 0x65, 0x64, 0x63, 0x61, 0x60, 0x5e, 0x5d, 0x5c, 0x5a, 0x59, 0x57, 0x56,
        0x54,
    ];

    Rgb::from([
        map_color(&COLORMAP_R, value),
        map_color(&COLORMAP_G, value),
        map_color(&COLORMAP_B, value),
    ])
}

#[inline]
fn map_color(colormap: &[u8; 256], value: f32) -> u8 {
    if value >= 1.0 {
        return colormap[colormap.len() - 1];
    }
    let step = 1.0 / (colormap.len() - 1) as f32;
    let box_index = ((value / step).floor() as usize).clamp(0, colormap.len() - 2);
    let ratio = (value - step * box_index as f32) / step;
    let c1 = colormap[box_index] as f32;
    let c2 = colormap[box_index + 1] as f32;
    (c2 * ratio + c1 * (1.0 - ratio)).round() as u8
}

#[derive(Debug)]
pub enum OutputError {
    Internal(&'static str),
    Io(std::io::Error),
    Image(image::ImageError),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Io(ref err) => err.fmt(f),
            Self::Image(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for OutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Io(ref err) => err.source(),
            Self::Image(ref err) => err.source(),
        }
    }
}

impl From<&'static str> for OutputError {
    fn from(msg: &'static str) -> OutputError {
        Self::Internal(msg)
    }
}

impl From<std::io::Error> for OutputError {
    fn from(e: std::io::Error) -> OutputError {
        Self::Io(e)
    }
}

impl From<image::ImageError> for OutputError {
    fn from(e: image::ImageError) -> OutputError {
        Self::Image(e)
    }
}
