use super::ConvBlock;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::{
        activation::{gelu, softmax},
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Tensor,
    },
};

const IMG_SIZE: usize = 384;
const PATCH_SIZE: usize = 16;
const EMBED_DIM: usize = 1024;

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    num_heads: usize,
    scale: f64,
}

struct AttentionConfig {
    dim: usize,
    num_heads: usize,
    qkv_bias: bool,
    proj_bias: bool,
}

impl AttentionConfig {
    fn init<B>(&self, device: &B::Device) -> Attention<B>
    where
        B: Backend,
    {
        let qkv = LinearConfig::new(self.dim, self.dim * 3)
            .with_bias(self.qkv_bias)
            .init(device);
        let proj = LinearConfig::new(self.dim, self.dim)
            .with_bias(self.proj_bias)
            .init(device);
        let scale = 1. / ((self.dim / self.num_heads) as f64).sqrt();
        Attention {
            qkv,
            proj,
            num_heads: self.num_heads,
            scale,
        }
    }
}

impl<B: Backend> Attention<B> {
    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, c] = xs.dims();
        let qkv = self
            .qkv
            .forward(xs)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let [q, k, v] = qkv
            .split(1, 0)
            .try_into()
            .expect("qkv should have 3 items in dimension 0");
        let q = q.squeeze(0).mul_scalar(self.scale);
        let k = k.squeeze::<4>(0);
        let v = v.squeeze::<4>(0);
        let attn = softmax(q.matmul(k.swap_dims(3, 2)), 3);
        let attn = attn.matmul(v.clone()).swap_dims(1, 2).reshape([b, n, c]);
        self.proj.forward(attn)
    }
}

#[derive(Module, Debug)]
struct LayerScale<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
}

impl<B> LayerScale<B>
where
    B: Backend,
{
    fn new(device: &B::Device, dim: usize) -> Self {
        let initializer = Initializer::Zeros;
        let gamma = initializer.init([dim], device);
        Self { gamma }
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        xs * self.gamma.val().unsqueeze_dims(&[0, 1])
    }
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B> Mlp<B>
where
    B: Backend,
{
    fn new(in_features: usize, hidden_features: usize, bias: bool, device: &B::Device) -> Mlp<B> {
        let out_features = in_features;
        let fc1 = LinearConfig::new(in_features, hidden_features)
            .with_bias(bias)
            .init(device);
        let fc2 = LinearConfig::new(hidden_features, out_features)
            .with_bias(bias)
            .init(device);
        Mlp { fc1, fc2 }
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let xs = self.fc1.forward(xs);
        let xs = gelu(xs);
        self.fc2.forward(xs)
    }
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    ls1: LayerScale<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
    ls2: LayerScale<B>,
}

impl<B> Block<B>
where
    B: Backend,
{
    fn new(dim: usize, num_heads: usize, device: &B::Device) -> Block<B> {
        let norm1 = LayerNormConfig::new(dim).init(device);
        let attn = AttentionConfig {
            dim,
            num_heads,
            qkv_bias: true,
            proj_bias: true,
        }
        .init(device);
        let ls1 = LayerScale::new(device, dim);
        let norm2 = LayerNormConfig::new(dim).init(device);
        let mlp = Mlp::new(dim, dim * 4, true, device);
        let ls2 = LayerScale::new(device, dim);
        Block {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        }
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = xs.clone();
        let xs = self.ls1.forward(self.attn.forward(self.norm1.forward(xs)));
        let xs = xs + residual;
        let residual = xs.clone();
        let xs = self.ls2.forward(self.mlp.forward(self.norm2.forward(xs)));
        xs + residual
    }
}

#[derive(Module, Debug)]
struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    patch_size: (usize, usize),
    num_patches: usize,
}

#[derive(Config, Debug)]
struct PatchEmbedConfig {
    img_size: usize,
    patch_size: usize,
    in_chans: usize,
    embed_dim: usize,
}

impl PatchEmbedConfig {
    fn init<B>(&self, device: &B::Device) -> PatchEmbed<B>
    where
        B: Backend,
    {
        let proj = Conv2dConfig::new(
            [self.in_chans, self.embed_dim],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.patch_size, self.patch_size])
        .init(device);
        let num_patches = (self.img_size / self.patch_size) * (self.img_size / self.patch_size);

        PatchEmbed {
            proj,
            patch_size: (self.patch_size, self.patch_size),
            num_patches,
        }
    }
}

impl<B: Backend> PatchEmbed<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 3> {
        let [_b, _c, h, w] = xs.dims();
        let (patch_h, patch_w) = self.patch_size;
        if (h % patch_h) != 0 {
            panic!("image height {h} is not a multiple of patch height {patch_h}");
        }
        if (w % patch_w) != 0 {
            panic!("image width {w} is not a multiple of patch width {patch_w}");
        }
        let xs = self.proj.forward(xs);
        let [b, c, h, w] = xs.dims();
        // flatten embeddings.
        xs.reshape([b, c, h * w]).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
struct DinoVisionTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: Param<Tensor<B, 3>>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
}

#[derive(Config, Debug)]
struct DinoVisionTransformerConfig {
    img_size: usize,
    patch_size: usize,
    depth: usize,
    embed_dim: usize,
    num_heads: usize,
}

impl DinoVisionTransformerConfig {
    pub fn init<B>(&self, device: &B::Device) -> DinoVisionTransformer<B>
    where
        B: Backend,
    {
        let patch_embed = PatchEmbedConfig {
            img_size: self.img_size,
            patch_size: self.patch_size,
            in_chans: 3,
            embed_dim: self.embed_dim,
        };
        let patch_embed = patch_embed.init(device);
        let initializer = Initializer::Zeros;
        let cls_token = initializer.init([1, 1, self.embed_dim], device);
        let num_tokens = 1;
        let pos_embed = initializer.init(
            [1, patch_embed.num_patches + num_tokens, self.embed_dim],
            device,
        );
        let norm = LayerNormConfig::new(self.embed_dim).init(device);
        let blocks = (0..self.depth)
            .map(|_i| Block::new(self.embed_dim, self.num_heads, device))
            .collect::<Vec<_>>();
        DinoVisionTransformer {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
        }
    }
}

impl<B: Backend> DinoVisionTransformer<B> {
    fn interpolate_pos_encoding(&self, xs_shape: [usize; 3], w: usize, h: usize) -> Tensor<B, 3> {
        let npatch = xs_shape[1] - 1;
        let n = self.pos_embed.dims()[1] - 1;
        if npatch != n || w != h {
            panic!("pos_embed interpolation is not implemented");
        }
        self.pos_embed.val()
    }

    fn prepare_tokens_with_mask(&self, xs: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _nc, w, h] = xs.dims();
        let xs = self.patch_embed.forward(xs);
        let cls_shape = self.cls_token.dims();
        let cls_token = self.cls_token.val().expand([b, cls_shape[1], cls_shape[2]]);
        let xs = Tensor::<B, 3>::cat(vec![cls_token, xs], 1);
        let xs_shape = xs.dims();
        xs + self.interpolate_pos_encoding(xs_shape, w, h)
    }

    fn get_intermediate_layers_not_chunked(
        &self,
        xs: Tensor<B, 4>,
        blocks_to_take: &[usize],
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let mut xs = self.prepare_tokens_with_mask(xs);
        let mut output = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            xs = blk.forward(xs);
            if blocks_to_take.contains(&i) {
                output.push(xs.clone());
            }
        }
        if output.len() != blocks_to_take.len() {
            panic!(
                "only {} / {} blocks found",
                output.len(),
                blocks_to_take.len()
            );
        }
        (xs, output)
    }

    fn forward_features(
        &self,
        xs: Tensor<B, 4>,
        intermediate_blocks: &[usize],
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        // Depth Pro uses forward_features instead of a normal forward call.
        // Based on vision_transformer.py.
        let (final_output, outputs) =
            self.get_intermediate_layers_not_chunked(xs, intermediate_blocks);
        let final_output = self.norm.forward(final_output);
        (final_output, outputs)
    }
}

fn dinov2l16_384_init<B: Backend>(device: &B::Device) -> DinoVisionTransformer<B> {
    let config = DinoVisionTransformerConfig {
        img_size: IMG_SIZE,
        patch_size: PATCH_SIZE,
        depth: 24,
        embed_dim: EMBED_DIM,
        num_heads: 16,
    };
    config.init(device)
}

#[derive(Module, Debug)]
pub(super) struct DepthProEncoder<B: Backend> {
    patch_encoder: DinoVisionTransformer<B>,
    image_encoder: DinoVisionTransformer<B>,
    upsample_latent0: Vec<ConvBlock<B>>,
    upsample_latent1: Vec<ConvBlock<B>>,
    upsample0: Vec<ConvBlock<B>>,
    upsample1: Vec<ConvBlock<B>>,
    upsample2: Vec<ConvBlock<B>>,
    upsample_lowres: ConvTranspose2d<B>,
    fuse_lowres: Conv2d<B>,
}

#[derive(Config, Debug)]
pub(super) struct DepthProEncoderConfig {}

impl DepthProEncoderConfig {
    pub fn init<B>(
        encoder_feature_dims: &[usize; 4],
        decoder_features: usize,
        device: &B::Device,
    ) -> DepthProEncoder<B>
    where
        B: Backend,
    {
        let patch_encoder = dinov2l16_384_init(device);
        let image_encoder = dinov2l16_384_init(device);
        let upsample_latent0 = Self::init_project_upsample_block(
            device,
            EMBED_DIM,
            decoder_features,
            3,
            Some(encoder_feature_dims[0]),
        );
        let upsample_latent1 =
            Self::init_project_upsample_block(device, EMBED_DIM, encoder_feature_dims[0], 2, None);
        let upsample0 =
            Self::init_project_upsample_block(device, EMBED_DIM, encoder_feature_dims[1], 1, None);
        let upsample1 =
            Self::init_project_upsample_block(device, EMBED_DIM, encoder_feature_dims[2], 1, None);
        let upsample2 =
            Self::init_project_upsample_block(device, EMBED_DIM, encoder_feature_dims[3], 1, None);
        let upsample_lowres =
            ConvTranspose2dConfig::new([EMBED_DIM, encoder_feature_dims[3]], [2, 2])
                .with_stride([2, 2])
                .init(device);
        let fuse_lowres = Conv2dConfig::new(
            [encoder_feature_dims[3] * 2, encoder_feature_dims[3]],
            [1, 1],
        )
        .init(device);
        DepthProEncoder {
            patch_encoder,
            image_encoder,
            upsample_latent0,
            upsample_latent1,
            upsample0,
            upsample1,
            upsample2,
            upsample_lowres,
            fuse_lowres,
        }
    }

    fn init_project_upsample_block<B>(
        device: &B::Device,
        dim_in: usize,
        dim_out: usize,
        upsample_layers: usize,
        dim_int: Option<usize>,
    ) -> Vec<ConvBlock<B>>
    where
        B: Backend,
    {
        let dim_int = dim_int.unwrap_or(dim_out);
        let mut layers = vec![ConvBlock {
            conv: Some(
                Conv2dConfig::new([dim_in, dim_int], [1, 1])
                    .with_bias(false)
                    .init(device),
            ),
            conv_tr: None,
        }];

        for i in 0..upsample_layers {
            let in_channels = if i == 0 { dim_int } else { dim_out };
            let layer = ConvTranspose2dConfig::new([in_channels, dim_out], [2, 2])
                .with_stride([2, 2])
                .with_bias(false)
                .init(device);
            layers.push(ConvBlock {
                conv: None,
                conv_tr: Some(layer),
            });
        }

        layers
    }
}

impl<B> DepthProEncoder<B>
where
    B: Backend,
{
    fn create_pyramid(x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // TODO: change to bicubic when not using Candle, or once Candle supports this.
        const INTERPOLATE_MODE: InterpolateMode = InterpolateMode::Nearest;
        let [_b, _c, h, w] = x.dims();
        let x1 = interpolate(
            x.clone(),
            [w / 2, h / 2],
            InterpolateOptions::new(INTERPOLATE_MODE),
        );
        let x2 = interpolate(
            x.clone(),
            [w / 4, h / 4],
            InterpolateOptions::new(INTERPOLATE_MODE),
        );
        let x0 = x;
        (x0, x1, x2)
    }

    fn split(x: Tensor<B, 4>, overlap_div: usize) -> Tensor<B, 4> {
        const PATCH_SIZE: usize = 384;
        let patch_stride = PATCH_SIZE - PATCH_SIZE / overlap_div;

        let image_size = x.dims()[3];

        let mut x_patch_list = vec![];
        for j in (0..=image_size - PATCH_SIZE).step_by(patch_stride) {
            let x_chunk = x.clone().narrow(2, j, PATCH_SIZE);
            for i in (0..=image_size - PATCH_SIZE).step_by(patch_stride) {
                x_patch_list.push(x_chunk.clone().narrow(3, i, PATCH_SIZE));
            }
        }
        Tensor::<B, 4>::cat(x_patch_list, 0)
    }

    fn merge(x: Tensor<B, 4>, batch_size: usize, padding: usize) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
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
                let output = x.clone().slice([b_range, 0..c, h_range, w_range]);
                output_row_list.push(output);
            }
            let output_row = Tensor::cat(output_row_list, 3);
            output_list.push(output_row);
        }
        Tensor::cat(output_list, 2)
    }

    fn reshape_feature(
        embeddings: Tensor<B, 3>,
        width: usize,
        height: usize,
        cls_token_offset: usize,
    ) -> Tensor<B, 4> {
        let [b, hw, c] = embeddings.dims();

        let embeddings = if cls_token_offset > 0 {
            embeddings.narrow(1, cls_token_offset, hw - cls_token_offset)
        } else {
            embeddings
        };

        embeddings
            .reshape([b, height, width, c])
            .permute([0, 3, 1, 2])
    }

    fn forward_seq(blocks: &[ConvBlock<B>], input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut result = input;
        for block in blocks {
            result = block.forward(result);
        }
        result
    }

    pub fn forward_encodings(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        const OUT_SIZE: usize = IMG_SIZE / PATCH_SIZE;
        const HIGHRES_LAYER_IDS: [usize; 2] = [5, 11];
        let batch_size = x.dims()[0];

        let (x0, x1, x2) = Self::create_pyramid(x.clone());

        let x0_patches = Self::split(x0, 4);
        let x1_patches = Self::split(x1, 2);
        let x2_patches = x2;
        let (x0_patches_len, x1_patches_len, x2_patches_len) = (
            x0_patches.dims()[0],
            x1_patches.dims()[0],
            x2_patches.dims()[0],
        );

        let x_pyramid_patches =
            Tensor::<B, 4>::cat(vec![x0_patches, x1_patches, x2_patches.clone()], 0);

        let (x_pyramid_encodings, highres_encodings) = self
            .patch_encoder
            .forward_features(x_pyramid_patches, &HIGHRES_LAYER_IDS);
        let [highres_encoding0, highres_encoding1] = highres_encodings
            .try_into()
            .expect("unexpected number of highres encodings");
        let x_pyramid_encodings = Self::reshape_feature(x_pyramid_encodings, OUT_SIZE, OUT_SIZE, 1);

        let x_latent0_encodings = Self::reshape_feature(highres_encoding0, OUT_SIZE, OUT_SIZE, 1);
        let x_latent0_features = Self::merge(
            x_latent0_encodings.narrow(0, 0, batch_size * 5 * 5),
            batch_size,
            3,
        );

        let x_latent1_encodings = Self::reshape_feature(highres_encoding1, OUT_SIZE, OUT_SIZE, 1);
        let x_latent1_features = Self::merge(
            x_latent1_encodings.narrow(0, 0, batch_size * 5 * 5),
            batch_size,
            3,
        );

        let [x0_encodings, x1_encodings, x2_encodings] = x_pyramid_encodings
            .split_with_sizes(vec![x0_patches_len, x1_patches_len, x2_patches_len], 0)
            .try_into()
            .expect("unexpected number of pyramid encodings");

        let x0_features = Self::merge(x0_encodings, batch_size, 3);
        let x1_features = Self::merge(x1_encodings, batch_size, 6);
        let x2_features = x2_encodings;

        let (x_global_features, _) = self.image_encoder.forward_features(x2_patches, &[]);
        let x_global_features = Self::reshape_feature(x_global_features, OUT_SIZE, OUT_SIZE, 1);

        let x_latent0_features = Self::forward_seq(&self.upsample_latent0, x_latent0_features);
        let x_latent1_features = Self::forward_seq(&self.upsample_latent1, x_latent1_features);

        let x0_features = Self::forward_seq(&self.upsample0, x0_features);
        let x1_features = Self::forward_seq(&self.upsample1, x1_features);
        let x2_features = Self::forward_seq(&self.upsample2, x2_features);

        let x_global_features = self.upsample_lowres.forward(x_global_features);
        let x_global_features = self
            .fuse_lowres
            .forward(Tensor::cat(vec![x2_features, x_global_features], 1));

        vec![
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
    }
}
