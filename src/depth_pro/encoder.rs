use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{
    layer_norm, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, Linear,
    Module, Sequential, VarBuilder,
};

const IMG_SIZE: usize = 384;
const PATCH_SIZE: usize = 16;
const EMBED_DIM: usize = 1024;

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        proj_bias: bool,
    ) -> Result<Self> {
        let qkv = linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = linear(vb.pp("proj"), dim, dim, proj_bias)?;
        let scale = 1. / ((dim / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)? // 02134
            .transpose(0, 1)? // 20134
            .transpose(2, 3)?; // 20314
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_features: usize, hidden_features: usize, bias: bool) -> Result<Self> {
        let out_features = in_features;
        let fc1 = linear(vb.pp("fc1"), in_features, hidden_features, bias)?;
        let fc2 = linear(vb.pp("fc2"), hidden_features, out_features, bias)?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&xs)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), dim, num_heads, true, true)?;
        let ls1 = LayerScale::new(vb.pp("ls1"), dim)?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, dim * 4, true)?;
        let ls2 = LayerScale::new(vb.pp("ls2"), dim)?;
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: Conv2d,
    patch_size: (usize, usize),
    num_patches: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size, config, vb.pp("proj"))?;
        let num_patches = (img_size / patch_size) * (img_size / patch_size);
        Ok(Self {
            proj,
            patch_size: (patch_size, patch_size),
            num_patches,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            candle_core::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle_core::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // flatten embeddings.
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct DinoVisionTransformer {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl DinoVisionTransformer {
    pub fn new(vb: VarBuilder, depth: usize, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let patch_embed =
            PatchEmbed::new(vb.pp("patch_embed"), IMG_SIZE, PATCH_SIZE, 3, embed_dim)?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let num_tokens = 1;
        let pos_embed = vb.get(
            (1, patch_embed.num_patches + num_tokens, embed_dim),
            "pos_embed",
        )?;
        let norm = layer_norm(embed_dim, 1e-5, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..depth)
            .map(|i| Block::new(vb_b.pp(i.to_string()), embed_dim, num_heads))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
        })
    }

    fn interpolate_pos_encoding(&self, xs: &Tensor, w: usize, h: usize) -> Result<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        let sqrt_n = (n as f64).sqrt();
        if npatch == n && w == h {
            return Ok(self.pos_embed.clone());
        }
        let class_pos_embed = self.pos_embed.i((.., ..1))?;
        let mut patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let (w0, h0) = ((w / PATCH_SIZE) as f64 + 0.1, (h / PATCH_SIZE) as f64 + 0.1);
        patch_pos_embed = patch_pos_embed
            .reshape((1, sqrt_n as usize, sqrt_n as usize, dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        // This uses bicubic interpolation in the original implementation.
        patch_pos_embed = patch_pos_embed.upsample_nearest2d(h0 as usize, w0 as usize)?;
        let el_count = patch_pos_embed.shape().elem_count();
        patch_pos_embed =
            patch_pos_embed
                .transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((1, el_count / dim, dim))?;
        Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)
    }

    fn prepare_tokens_with_mask(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _nc, w, h) = xs.dims4()?;
        let mut xs = self.patch_embed.forward(xs)?;
        let cls_shape = self.cls_token.dims3()?;
        let cls_token = self.cls_token.expand((b, cls_shape.1, cls_shape.2))?;
        xs = Tensor::cat(&[&cls_token, &xs], 1)?;
        xs.broadcast_add(&self.interpolate_pos_encoding(&xs, w, h)?)
    }

    fn get_intermediate_layers_not_chunked(
        &self,
        xs: &Tensor,
        blocks_to_take: &[usize],
    ) -> Result<Vec<Tensor>> {
        let mut xs = self.prepare_tokens_with_mask(xs)?;
        let mut output = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            xs = blk.forward(&xs)?;
            if blocks_to_take.contains(&i) {
                output.push(xs.clone());
            }
        }
        if output.len() != blocks_to_take.len() {
            candle_core::bail!(
                "only {} / {} blocks found",
                output.len(),
                blocks_to_take.len()
            );
        }
        Ok(output)
    }

    fn forward_features(
        &self,
        xs: &Tensor,
        intermediate_blocks: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        // Depth Pro uses forward_features instead of a normal forward call.
        let mut blocks_to_take = intermediate_blocks.to_vec();
        blocks_to_take.push(self.blocks.len() - 1);
        // Based on vision_transformer.py.
        let mut outputs = self.get_intermediate_layers_not_chunked(xs, &blocks_to_take)?;
        let mut final_output = if let Some(xs) = outputs.pop() {
            xs
        } else {
            candle_core::bail!("failed to return last block when forwarding features")
        };
        final_output = self.norm.forward(&final_output)?;
        Ok((final_output, outputs))
    }
}

fn dinov2l16_384(vb: VarBuilder) -> Result<DinoVisionTransformer> {
    DinoVisionTransformer::new(vb, 24, EMBED_DIM, 16)
}

pub(super) struct DepthProEncoder {
    patch_encoder: DinoVisionTransformer,
    image_encoder: DinoVisionTransformer,
    upsample_latent0: Sequential,
    upsample_latent1: Sequential,
    upsample0: Sequential,
    upsample1: Sequential,
    upsample2: Sequential,
    upsample_lowres: ConvTranspose2d,
    fuse_lowres: Conv2d,
}

impl DepthProEncoder {
    pub fn new(
        vb: VarBuilder,
        encoder_feature_dims: &[usize],
        decoder_features: usize,
    ) -> Result<DepthProEncoder> {
        if encoder_feature_dims.len() != 4 {
            let l = encoder_feature_dims.len();
            candle_core::bail!("expected 4 encoder feature dims, got {l}");
        }
        let patch_encoder = dinov2l16_384(vb.pp("patch_encoder"))?;
        let image_encoder = dinov2l16_384(vb.pp("image_encoder"))?;
        let upsample_latent0 = Self::create_project_upsample_block(
            EMBED_DIM,
            decoder_features,
            3,
            Some(encoder_feature_dims[0]),
            vb.pp("upsample_latent0"),
        )?;
        let upsample_latent1 = Self::create_project_upsample_block(
            EMBED_DIM,
            encoder_feature_dims[0],
            2,
            None,
            vb.pp("upsample_latent1"),
        )?;
        let upsample0 = Self::create_project_upsample_block(
            EMBED_DIM,
            encoder_feature_dims[1],
            1,
            None,
            vb.pp("upsample0"),
        )?;
        let upsample1 = Self::create_project_upsample_block(
            EMBED_DIM,
            encoder_feature_dims[2],
            1,
            None,
            vb.pp("upsample1"),
        )?;
        let upsample2 = Self::create_project_upsample_block(
            EMBED_DIM,
            encoder_feature_dims[3],
            1,
            None,
            vb.pp("upsample2"),
        )?;
        let upsample_lowres = candle_nn::conv_transpose2d(
            EMBED_DIM,
            encoder_feature_dims[3],
            2,
            ConvTranspose2dConfig {
                padding: 0,
                output_padding: 0,
                stride: 2,
                dilation: 1,
            },
            vb.pp("upsample_lowres"),
        )?;
        let fuse_lowres = candle_nn::conv2d(
            encoder_feature_dims[3] * 2,
            encoder_feature_dims[3],
            1,
            Conv2dConfig::default(),
            vb.pp("fuse_lowres"),
        )?;
        Ok(DepthProEncoder {
            patch_encoder,
            image_encoder,
            upsample_latent0,
            upsample_latent1,
            upsample0,
            upsample1,
            upsample2,
            upsample_lowres,
            fuse_lowres,
        })
    }

    fn create_project_upsample_block(
        dim_in: usize,
        dim_out: usize,
        upsample_layers: usize,
        dim_int: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Sequential> {
        let dim_int = dim_int.unwrap_or(dim_out);
        let mut layer = candle_nn::seq().add(candle_nn::conv2d_no_bias(
            dim_in,
            dim_int,
            1,
            Conv2dConfig::default(),
            vb.pp(0),
        )?);

        for i in 0..upsample_layers {
            let in_channels = if i == 0 { dim_int } else { dim_out };
            let cfg = ConvTranspose2dConfig {
                padding: 0,
                output_padding: 0,
                stride: 2,
                dilation: 1,
            };
            layer = layer.add(candle_nn::conv_transpose2d_no_bias(
                in_channels,
                dim_out,
                2,
                cfg,
                vb.pp(1 + i),
            )?);
        }

        Ok(layer)
    }

    fn create_pyramid(x: Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (_b, _c, h, w) = x.dims4()?;
        // Depth Pro uses bicubic interpolation, which is not supported by Candle at the moment.
        // Seems unnecessary, as bicubic doesn't seem to be the best algorithm downscaling
        // (especially by 2^x).
        let x1 = x.interpolate2d(w / 2, h / 2)?;
        let x2 = x.interpolate2d(w / 4, h / 4)?;
        let x0 = x;
        Ok((x0, x1, x2))
    }

    fn split(x: Tensor, overlap_div: usize) -> Result<Tensor> {
        const PATCH_SIZE: usize = 384;
        let patch_stride = PATCH_SIZE - PATCH_SIZE / overlap_div;

        let image_size = x.dims4()?.3;

        let mut x_patch_list = vec![];
        for j in (0..=image_size - PATCH_SIZE).step_by(patch_stride) {
            for i in (0..=image_size - PATCH_SIZE).step_by(patch_stride) {
                x_patch_list.push(x.i((.., .., j..j + PATCH_SIZE, i..i + PATCH_SIZE))?);
            }
        }
        Tensor::cat(&x_patch_list, 0)
    }

    fn merge(x: Tensor, batch_size: usize, padding: usize) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;
        let steps = ((b / batch_size) as f64).sqrt() as usize;

        let mut output_list = vec![];
        for j in 0..steps {
            let mut output_row_list = vec![];
            for i in 0..steps {
                let idx = j * steps + i;
                let b_range = batch_size * idx..batch_size * (idx + 1);
                let mut h_range = 0..h;
                let mut w_range = 0..w;
                if j > 0 {
                    h_range.start = padding;
                }
                if i > 0 {
                    w_range.start = padding;
                }
                if j < steps - 1 {
                    h_range.end = h - padding;
                }
                if i < steps - 1 {
                    w_range.end = w - padding;
                }
                let output = x.i((b_range, .., h_range, w_range))?;
                output_row_list.push(output);
            }
            let output_row = Tensor::cat(&output_row_list, D::Minus1)?;
            output_list.push(output_row);
        }
        Tensor::cat(&output_list, D::Minus2)
    }

    fn reshape_feature(
        embeddings: Tensor,
        width: usize,
        height: usize,
        cls_token_offset: usize,
    ) -> Result<Tensor> {
        let (b, _hw, c) = embeddings.dims3()?;

        let embeddings = if cls_token_offset > 0 {
            embeddings.i((.., cls_token_offset.., ..))?
        } else {
            embeddings
        };

        embeddings
            .reshape(&[b, height, width, c])?
            .permute((0, 3, 1, 2))
    }

    pub fn forward_encodings(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        const OUT_SIZE: usize = IMG_SIZE / PATCH_SIZE;
        const HIGHRES_LAYER_IDS: [usize; 2] = [5, 11];
        let batch_size = x.dim(0)?;

        let (x0, x1, x2) = Self::create_pyramid(x.clone())?;

        let x0_patches = Self::split(x0, 4)?;
        let x1_patches = Self::split(x1, 2)?;
        let x2_patches = x2;
        let (x0_patches_len, x1_patches_len, x2_patches_len) = (
            x0_patches.dims4()?.0,
            x1_patches.dims4()?.0,
            x2_patches.dims4()?.0,
        );

        let x_pyramid_patches = Tensor::cat(&[x0_patches, x1_patches, x2_patches.clone()], 0)?;

        println!("forward start");
        let (x_pyramid_encodings, highres_encodings) = self
            .patch_encoder
            .forward_features(&x_pyramid_patches, &HIGHRES_LAYER_IDS)?;
        println!("forward end");
        drop(x_pyramid_patches);
        let (highres_encoding0, highres_encoding1) = {
            let mut it = highres_encodings.into_iter();
            (it.next().take().unwrap(), it.next().take().unwrap())
        };
        let x_pyramid_encodings =
            Self::reshape_feature(x_pyramid_encodings, OUT_SIZE, OUT_SIZE, 1)?;

        let x_latent0_encodings = Self::reshape_feature(highres_encoding0, OUT_SIZE, OUT_SIZE, 1)?;
        let x_latent0_features =
            Self::merge(x_latent0_encodings.i(..batch_size * 5 * 5)?, batch_size, 3)?;
        drop(x_latent0_encodings);

        let x_latent1_encodings = Self::reshape_feature(highres_encoding1, OUT_SIZE, OUT_SIZE, 1)?;
        let x_latent1_features =
            Self::merge(x_latent1_encodings.i(..batch_size * 5 * 5)?, batch_size, 3)?;
        drop(x_latent1_encodings);

        let x0_encodings = x_pyramid_encodings.i(..x0_patches_len)?;
        let x1_encodings = x_pyramid_encodings.i(x0_patches_len..x0_patches_len + x1_patches_len)?;
        let x2_encodings = x_pyramid_encodings
            .i(x0_patches_len + x1_patches_len..x0_patches_len + x1_patches_len + x2_patches_len)?;
        drop(x_pyramid_encodings);

        let mut x0_features = Self::merge(x0_encodings, batch_size, 3)?;
        let mut x1_features = Self::merge(x1_encodings, batch_size, 6)?;
        let mut x2_features = x2_encodings;

        println!("forward start");
        let (mut x_global_features, _) = self.image_encoder.forward_features(&x2_patches, &[])?;
        println!("forward end");
        drop(x2_patches);
        x_global_features = Self::reshape_feature(x_global_features, OUT_SIZE, OUT_SIZE, 1)?;

        let x_latent0_features = self.upsample_latent0.forward(&x_latent0_features)?;
        let x_latent1_features = self.upsample_latent1.forward(&x_latent1_features)?;

        x0_features = self.upsample0.forward(&x0_features)?;
        x1_features = self.upsample1.forward(&x1_features)?;
        x2_features = self.upsample2.forward(&x2_features)?;

        x_global_features = self.upsample_lowres.forward(&x_global_features)?;
        x_global_features = self
            .fuse_lowres
            .forward(&Tensor::cat(&[x2_features, x_global_features], 1)?)?;

        Ok(vec![
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ])
    }
}
