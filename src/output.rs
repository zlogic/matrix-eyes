use std::{
    fmt,
    fs::File,
    io::{BufWriter, Write as _},
    ops::Range,
    path::Path,
};

use burn::{
    prelude::Backend,
    tensor::{DataError, Tensor},
};
use image::{
    DynamicImage, ImageReader, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use rand::Rng as _;

pub struct DepthMap {
    data: Vec<f32>,
    data_width: usize,
    data_height: usize,
    original_width: u32,
    original_height: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageOutputFormat {
    DepthMap,
    Stereogram(Option<f32>, f32),
}

#[derive(Debug, Clone, Copy)]
pub enum VertexMode {
    Plain,
    Color,
    Texture,
}

const POLYGON_DEPTH_THRESHOLD: f32 = 1.025;
const CLIP_DEPTH_RANGE: Range<f32> = 0.1..250.0;

impl DepthMap {
    pub fn new<B>(
        inverse_depth: Tensor<B, 2>,
        original_size: (u32, u32),
    ) -> Result<DepthMap, DataError>
    where
        B: Backend,
    {
        const CLAMP_RANGE: Range<f32> = 1.0 / CLIP_DEPTH_RANGE.end..1.0 / CLIP_DEPTH_RANGE.start;
        let [data_width, data_height] = inverse_depth.dims();
        let mut data = inverse_depth.to_data().to_vec()?;
        data.iter_mut()
            .for_each(|v: &mut f32| *v = v.clamp(CLAMP_RANGE.start, CLAMP_RANGE.end));
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
        source_path: &str,
        image_format: ImageOutputFormat,
        vertex_mode: VertexMode,
    ) -> Result<(), OutputError> {
        if destination_path.to_lowercase().as_str().ends_with(".ply") {
            let writer = PlyWriter::new(destination_path, vertex_mode)?;
            self.output_mesh(source_path, writer, vertex_mode)
        } else if destination_path.to_lowercase().as_str().ends_with(".obj") {
            let writer = ObjWriter::new(destination_path, source_path, vertex_mode)?;
            self.output_mesh(source_path, writer, vertex_mode)
        } else {
            match image_format {
                ImageOutputFormat::DepthMap => self.output_depth_map(destination_path),
                ImageOutputFormat::Stereogram(resize_scale, amplitude) => {
                    self.output_stereogram(destination_path, resize_scale, amplitude)
                }
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
        let pattern_width = (depth_multiplier * 2.0 + amplitude).round() as usize;

        let mut rng = rand::rng();
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

    fn output_mesh<M>(
        &self,
        source_path: &str,
        mut writer: M,
        mode: VertexMode,
    ) -> Result<(), OutputError>
    where
        M: MeshWriter,
    {
        let indexed_mesh = IndexedMesh::new(&self.data, self.data_width, self.data_height);

        let original_image = if matches!(mode, VertexMode::Color) {
            let img = ImageReader::open(source_path)?
                .decode()?
                .resize_exact(
                    self.data_width as u32,
                    self.data_height as u32,
                    FilterType::Lanczos3,
                )
                .into_rgb8();
            Some(img)
        } else {
            None
        };

        // When resizing image, the larger dimension will be squished.
        // To restore original coordinates, multiply by its squish factor.
        let x_multiplier =
            self.original_width as f32 / self.original_width.max(self.original_height) as f32;
        let y_multiplier =
            self.original_height as f32 / self.original_width.max(self.original_height) as f32;

        writer.output_header(indexed_mesh.nvertices, indexed_mesh.nfaces)?;
        for (x_image, y_image, _i, _v) in indexed_mesh.sorted_vertices().iter() {
            let (x_norm, y_norm) = (
                *x_image as f32 / self.data_width as f32,
                *y_image as f32 / self.data_height as f32,
            );
            writer.output_vertex_uv(x_norm, y_norm)?;
        }
        for (x_image, y_image, i, _v) in indexed_mesh.sorted_vertices().into_iter() {
            let color = original_image.as_ref().and_then(|img| {
                img.get_pixel_checked(x_image as u32, y_image as u32)
                    .map(|pixel| pixel.0)
            });

            let (x_norm, y_norm) = (
                x_image as f32 / self.data_width as f32,
                y_image as f32 / self.data_height as f32,
            );
            let z_norm = 1.0 / self.data[i];
            let x = x_multiplier * (x_norm - 0.5) * z_norm;
            let y = y_multiplier * (y_norm - 0.5) * z_norm;
            writer.output_vertex(x, y, z_norm, color)?;
        }

        IndexedMesh::for_each_face(&self.data, self.data_width, self.data_height, |vertices| {
            if let Some(face) = indexed_mesh.remap_face(vertices) {
                writer.output_face(face)?;
            }
            Ok::<(), OutputError>(())
        })?;

        writer.complete()?;

        Ok(())
    }
}

struct IndexedMesh {
    vertices: Vec<Option<usize>>,
    width: usize,
    nvertices: usize,
    nfaces: usize,
}

impl IndexedMesh {
    fn new(vertices: &[f32], width: usize, height: usize) -> IndexedMesh {
        let mut index = vec![None; vertices.len()];
        let mut next_vertex = 0usize;
        let mut faces_count = 0usize;
        Self::for_each_face(vertices, width, height, |indices| {
            indices.iter().for_each(|i| {
                index[*i].get_or_insert_with(|| {
                    let value = next_vertex;
                    next_vertex += 1;
                    value
                });
            });
            faces_count += 1;
            Ok::<(), OutputError>(())
        })
        .expect("unexpected error when indexing vertices");
        IndexedMesh {
            vertices: index,
            width,
            nvertices: next_vertex,
            nfaces: faces_count,
        }
    }

    fn sorted_vertices(&self) -> Vec<(usize, usize, usize, usize)> {
        let mut vertices = self
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, v)| Some((i % self.width, i / self.width, i, (*v)?)))
            .collect::<Vec<_>>();
        vertices.sort_by_key(|(_x, _y, _i, v)| *v);
        vertices
    }

    fn for_each_face<F, E>(
        vertices: &[f32],
        width: usize,
        height: usize,
        mut process_face: F,
    ) -> Result<(), E>
    where
        F: FnMut([usize; 3]) -> Result<(), E>,
    {
        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let i00 = y * width + x;
                let i10 = y * width + x + 1;
                let i01 = (y + 1) * width + x;
                let i11 = (y + 1) * width + x + 1;

                let v00 = vertices[i00];
                let v10 = vertices[i10];
                let v01 = vertices[i01];
                let v11 = vertices[i11];

                let i_upper_left = [i00, i01, i10];
                let i_lower_right = [i10, i01, i11];
                let v_upper_left = [v00, v01, v10];
                let v_lower_right = [v10, v01, v11];

                if let (Some(min), Some(max)) = (
                    v_upper_left.iter().min_by(|a, b| a.total_cmp(b)),
                    v_upper_left.iter().max_by(|a, b| a.total_cmp(b)),
                ) {
                    // TODO: use absolute scale here? or at least something that works around +- zero?
                    if max / min <= POLYGON_DEPTH_THRESHOLD {
                        process_face(i_upper_left)?;
                    }
                }

                if let (Some(min), Some(max)) = (
                    v_lower_right.iter().min_by(|a, b| a.total_cmp(b)),
                    v_lower_right.iter().max_by(|a, b| a.total_cmp(b)),
                ) {
                    // TODO: use absolute scale here? or at least something that works around +- zero?
                    if max / min <= POLYGON_DEPTH_THRESHOLD {
                        process_face(i_lower_right)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn remap_face(&self, original_indices: [usize; 3]) -> Option<[usize; 3]> {
        let i0 = self.vertices[original_indices[0]]?;
        let i1 = self.vertices[original_indices[1]]?;
        let i2 = self.vertices[original_indices[2]]?;
        Some([i0, i1, i2])
    }
}

trait MeshWriter {
    fn output_header(&mut self, nvertices: usize, nfaces: usize) -> Result<(), std::io::Error>;

    fn output_vertex(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        color: Option<[u8; 3]>,
    ) -> Result<(), OutputError>;

    fn output_vertex_uv(&mut self, u: f32, v: f32) -> Result<(), OutputError>;

    fn output_face(&mut self, indices: [usize; 3]) -> Result<(), OutputError>;

    fn complete(&mut self) -> Result<(), OutputError>;
}

const WRITE_BUFFER_SIZE: usize = 1024 * 1024;

struct PlyWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    vertex_mode: VertexMode,
}

impl PlyWriter {
    fn new(path: &str, vertex_mode: VertexMode) -> Result<PlyWriter, OutputError> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        Ok(PlyWriter {
            writer,
            buffer,
            vertex_mode,
        })
    }

    fn check_flush_buffer(&mut self) -> Result<(), std::io::Error> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        if buffer.len() >= WRITE_BUFFER_SIZE {
            w.write_all(buffer)?;
            buffer.clear();
        }
        Ok(())
    }
}

impl MeshWriter for PlyWriter {
    fn output_header(&mut self, nvertices: usize, nfaces: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;
        writeln!(w, "ply")?;
        writeln!(w, "format binary_big_endian 1.0")?;
        writeln!(w, "comment Matrix Eyes 3D surface")?;
        writeln!(w, "element vertex {nvertices}")?;
        writeln!(w, "property double x")?;
        writeln!(w, "property double y")?;
        writeln!(w, "property double z")?;

        match self.vertex_mode {
            VertexMode::Plain => {}
            VertexMode::Texture => {}
            VertexMode::Color => {
                writeln!(w, "property uchar red")?;
                writeln!(w, "property uchar green")?;
                writeln!(w, "property uchar blue")?;
            }
        }
        writeln!(w, "element face {nfaces}")?;
        writeln!(w, "property list uchar int vertex_indices")?;
        writeln!(w, "end_header")
    }

    fn output_vertex(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        color: Option<[u8; 3]>,
    ) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let (x, y, z) = (x as f64, -y as f64, -z as f64);
        w.write_all(&x.to_be_bytes())?;
        w.write_all(&y.to_be_bytes())?;
        w.write_all(&z.to_be_bytes())?;
        if let Some(color) = color {
            w.write_all(&color)?;
        };
        Ok(())
    }

    fn output_vertex_uv(&mut self, _u: f32, _v: f32) -> Result<(), OutputError> {
        Ok(())
    }

    fn output_face(&mut self, indices: [usize; 3]) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;
        const NUM_POINTS: [u8; 1] = 3u8.to_be_bytes();
        w.write_all(&NUM_POINTS)?;
        w.write_all(&(indices[0] as u32).to_be_bytes())?;
        w.write_all(&(indices[1] as u32).to_be_bytes())?;
        w.write_all(&(indices[2] as u32).to_be_bytes())?;
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        Ok(())
    }
}

struct ObjWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    vertex_mode: VertexMode,
    path: String,
    image_path: String,
}

impl ObjWriter {
    fn new(
        path: &str,
        image_path: &str,
        vertex_mode: VertexMode,
    ) -> Result<ObjWriter, OutputError> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
        Ok(ObjWriter {
            writer,
            buffer,
            vertex_mode,
            path: path.to_string(),
            image_path: image_path.to_string(),
        })
    }

    fn check_flush_buffer(&mut self) -> Result<(), std::io::Error> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        if buffer.len() >= WRITE_BUFFER_SIZE {
            w.write_all(buffer)?;
            buffer.clear();
        }
        Ok(())
    }

    fn get_output_filename(&self) -> Option<String> {
        Path::new(&self.path)
            .file_stem()
            .and_then(|n| n.to_str().map(|n| n.to_string()))
    }

    fn write_materials(&mut self) -> Result<(), OutputError> {
        let out_filename = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => return Ok(()),
            VertexMode::Texture => self.get_output_filename().unwrap(),
        };

        let destination_path = Path::new(&self.path).parent().unwrap();
        let mut w = BufWriter::new(File::create(
            destination_path.join(format!("{out_filename}.mtl")),
        )?);

        writeln!(w, "newmtl Textured")?;
        writeln!(w, "Ka 0.2 0.2 0.2")?;
        writeln!(w, "Kd 0.8 0.8 0.8")?;
        writeln!(w, "Ks 1.0 1.0 1.0")?;
        writeln!(w, "illum 2")?;
        writeln!(w, "Ns 0.000500")?;
        writeln!(w, "map_Ka {}", self.image_path)?;
        writeln!(w, "map_Kd {}", self.image_path)?;
        writeln!(w)?;

        Ok(())
    }
}

impl MeshWriter for ObjWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let out_filename = self.get_output_filename().unwrap();
        let w = &mut self.buffer;

        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                writeln!(w, "mtllib {out_filename}.mtl")?;
                writeln!(w, "usemtl Textured")?;
            }
        }
        Ok(())
    }

    fn output_vertex(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        color: Option<[u8; 3]>,
    ) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let (x, y, z) = (x as f64, -y as f64, -z as f64);
        write!(w, "v {x} {y} {z}")?;
        if let Some(color) = color {
            write!(
                w,
                " {} {} {}",
                color[0] as f64 / 255.0,
                color[1] as f64 / 255.0,
                color[2] as f64 / 255.0,
            )?
        }
        writeln!(w)?;

        Ok(())
    }

    fn output_vertex_uv(&mut self, u: f32, v: f32) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                let w = &mut self.buffer;
                writeln!(w, "vt {} {}", u as f64, 1.0f64 - v as f64)?
            }
        }
        Ok(())
    }

    fn output_face(&mut self, indices: [usize; 3]) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        write!(self.buffer, "f")?;
        for index in indices {
            let index = index + 1;
            match self.vertex_mode {
                VertexMode::Plain | VertexMode::Color => {
                    write!(self.buffer, " {index}")?;
                }
                VertexMode::Texture => {
                    write!(self.buffer, " {index}/{index}")?;
                }
            }
        }
        writeln!(self.buffer)?;
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        self.write_materials()?;
        Ok(())
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
