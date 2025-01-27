use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::{
        activation::{gelu, softmax},
        Tensor,
    },
};

pub(super) const IMG_SIZE: usize = 384;
pub(super) const PATCH_SIZE: usize = 16;
pub(super) const EMBED_DIM: usize = 1024;

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
pub(super) struct DinoVisionTransformer<B: Backend> {
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

    pub fn forward_features(
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

pub(super) fn dinov2l16_384_init<B: Backend>(device: &B::Device) -> DinoVisionTransformer<B> {
    let config = DinoVisionTransformerConfig {
        img_size: IMG_SIZE,
        patch_size: PATCH_SIZE,
        depth: 24,
        embed_dim: EMBED_DIM,
        num_heads: 16,
    };
    config.init(device)
}
