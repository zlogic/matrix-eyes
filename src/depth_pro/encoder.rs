use super::{
    ConvBlock, ProgressListener, SplitProgressListener,
    vit::{self, DinoVisionTransformer},
};
use burn::{
    config::Config,
    module::Module,
    nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    prelude::Backend,
    tensor::{
        Tensor,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

const IMG_SIZE: usize = vit::IMG_SIZE;
const EMBED_DIM: usize = vit::EMBED_DIM;
const PATCH_SIZE: usize = vit::PATCH_SIZE;

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
        let patch_encoder = vit::dinov2l16_384_init(device);
        let image_encoder = vit::dinov2l16_384_init(device);
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
        const INTERPOLATE_MODE: InterpolateMode = if cfg!(feature = "candle-cuda") {
            InterpolateMode::Nearest
        } else {
            InterpolateMode::Bilinear
        };
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

    pub fn forward_encodings<PL>(
        &self,
        x: Tensor<B, 4>,
        pl: SplitProgressListener<PL>,
    ) -> Vec<Tensor<B, 4>>
    where
        PL: ProgressListener,
    {
        const OUT_SIZE: usize = IMG_SIZE / PATCH_SIZE;
        const HIGHRES_LAYER_IDS: [usize; 2] = [5, 11];
        let batch_size = x.dims()[0];

        let (pl, pl_image) = pl.split_range(0.5);

        let (pl, pl_next) = pl.split_range(0.02);
        pl.update_message("creating image pyramid".into());
        let (x0, x1, x2) = Self::create_pyramid(x.clone());
        pl.report_status(0.25);

        pl.update_message("preparing image patches".into());
        let x0_patches = Self::split(x0, 4);
        pl.report_status(0.45);
        let x1_patches = Self::split(x1, 2);
        pl.report_status(0.65);
        let x2_patches = x2;
        let (x0_patches_len, x1_patches_len, x2_patches_len) = (
            x0_patches.dims()[0],
            x1_patches.dims()[0],
            x2_patches.dims()[0],
        );

        let x_pyramid_patches =
            Tensor::<B, 4>::cat(vec![x0_patches, x1_patches, x2_patches.clone()], 0);

        pl.update_message("encoding patches".into());
        let (pl_features, pl) = pl_next.split_range(0.95);
        let (x_pyramid_encodings, highres_encodings) =
            self.patch_encoder
                .forward_features(x_pyramid_patches, &HIGHRES_LAYER_IDS, pl_features);

        pl.update_message("reshaping patch encodings".into());
        pl.report_status(0.0);
        let [highres_encoding0, highres_encoding1] = highres_encodings
            .try_into()
            .expect("unexpected number of highres encodings");
        let x_pyramid_encodings = Self::reshape_feature(x_pyramid_encodings, OUT_SIZE, OUT_SIZE, 1);

        pl.report_status(0.2);
        let x_latent0_encodings = Self::reshape_feature(highres_encoding0, OUT_SIZE, OUT_SIZE, 1);
        let x_latent0_features = Self::merge(
            x_latent0_encodings.narrow(0, 0, batch_size * 5 * 5),
            batch_size,
            3,
        );

        pl.report_status(0.4);
        let x_latent1_encodings = Self::reshape_feature(highres_encoding1, OUT_SIZE, OUT_SIZE, 1);
        pl.report_status(0.6);
        let x_latent1_features = Self::merge(
            x_latent1_encodings.narrow(0, 0, batch_size * 5 * 5),
            batch_size,
            3,
        );

        let (pl, pl_next) = pl_image.split_range(0.02);
        pl.update_message("preparing image encodings".into());
        pl.report_status(0.0);
        let [x0_encodings, x1_encodings, x2_encodings] = x_pyramid_encodings
            .split_with_sizes(vec![x0_patches_len, x1_patches_len, x2_patches_len], 0)
            .try_into()
            .expect("unexpected number of pyramid encodings");
        pl.report_status(0.5);

        let x0_features = Self::merge(x0_encodings, batch_size, 3);
        pl.report_status(0.75);
        let x1_features = Self::merge(x1_encodings, batch_size, 6);
        let x2_features = x2_encodings;

        pl.update_message("encoding image".into());
        let (pl_image, pl) = pl_next.split_range(0.1);
        let (x_global_features, _) = self
            .image_encoder
            .forward_features(x2_patches, &[], pl_image);

        pl.update_message("reshaping image encodings".into());
        let x_global_features = Self::reshape_feature(x_global_features, OUT_SIZE, OUT_SIZE, 1);
        pl.report_status(0.1);

        pl.update_message("encoding features".into());
        let x_latent0_features = Self::forward_seq(&self.upsample_latent0, x_latent0_features);
        pl.report_status(0.3);
        let x_latent1_features = Self::forward_seq(&self.upsample_latent1, x_latent1_features);
        pl.report_status(0.5);

        let x0_features = Self::forward_seq(&self.upsample0, x0_features);
        pl.report_status(0.6);
        let x1_features = Self::forward_seq(&self.upsample1, x1_features);
        pl.report_status(0.7);
        let x2_features = Self::forward_seq(&self.upsample2, x2_features);
        pl.report_status(0.8);

        pl.update_message("upsampling lowres".into());
        let x_global_features = self.upsample_lowres.forward(x_global_features);
        pl.report_status(0.9);
        pl.update_message("fusing lowres".into());
        let x_global_features = self
            .fuse_lowres
            .forward(Tensor::cat(vec![x2_features, x_global_features], 1));
        pl.report_status(1.0);

        vec![
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]
    }
}
