use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
    tensor::{
        Tensor,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use super::{
    ProgressListener, SplitProgressListener,
    vit::{self, DinoVisionTransformer},
};

const EMBED_DIM: usize = vit::EMBED_DIM;

#[derive(Module, Debug)]
struct Encoder<B: Backend> {
    fov_encoder: DinoVisionTransformer<B>,
    linear: Linear<B>,
}

#[derive(Module, Debug)]
pub(super) struct FOVNetwork<B: Backend> {
    encoder: Encoder<B>,
    downsample: Vec<Conv2d<B>>,
    head: Vec<Conv2d<B>>,
}

impl<B> FOVNetwork<B>
where
    B: Backend,
{
    pub fn forward<PL>(
        &self,
        x: Tensor<B, 4>,
        lowres_feature: Tensor<B, 4>,
        pl: SplitProgressListener<PL>,
    ) -> Tensor<B, 1>
    where
        PL: ProgressListener,
    {
        const INTERPOLATE_MODE: InterpolateMode = if cfg!(feature = "candle-cuda") {
            InterpolateMode::Nearest
        } else {
            InterpolateMode::Bilinear
        };
        let (pl, pl_next) = pl.split_range(0.01);
        let [_b, _c, h, w] = x.dims();
        pl.update_message("interpolating image".into());
        let x = interpolate(x, [w / 4, h / 4], InterpolateOptions::new(INTERPOLATE_MODE));
        pl.report_status(1.0);
        pl.update_message("encoding fov".into());
        let (pl_encoder, pl) = pl_next.split_range(0.8);
        let x = self
            .encoder
            .fov_encoder
            .forward_features(x, &[], pl_encoder)
            .0;
        pl.update_message("fov linear".into());
        let x = self.encoder.linear.forward(x);
        pl.report_status(0.4);

        let [_, x_dim1, _] = x.dims();
        let x = x.narrow(1, 1, x_dim1 - 1).permute([0, 2, 1]);

        pl.update_message("fov lowres".into());
        let lowres_feature = self.downsample[0].forward(lowres_feature);
        pl.report_status(0.65);
        let lowres_feature = Relu::new().forward(lowres_feature);
        pl.report_status(0.70);
        let x = x.reshape(lowres_feature.dims()) + lowres_feature;
        pl.report_status(0.75);

        let x = self.head[0].forward(x);
        pl.report_status(0.80);
        let x = Relu::new().forward(x);
        pl.report_status(0.85);
        let x = self.head[1].forward(x);
        pl.report_status(0.90);
        let x = Relu::new().forward(x);
        pl.report_status(0.95);
        let x = self.head[2].forward(x);

        x.squeeze_dims(&[0, 1, 2])
    }
}

#[derive(Config, Debug)]
pub(super) struct FOVNetworkConfig {}

impl FOVNetworkConfig {
    pub fn init<B>(num_features: usize, device: &B::Device) -> FOVNetwork<B>
    where
        B: Backend,
    {
        let fov_encoder = vit::dinov2l16_384_init(device);

        let fov_head0 = Conv2dConfig::new([num_features, num_features / 2], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let fov_head = vec![
            Conv2dConfig::new([num_features / 2, num_features / 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            Conv2dConfig::new([num_features / 4, num_features / 8], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            Conv2dConfig::new([num_features / 8, 1], [6, 6]).init(device),
        ];

        let encoder = Encoder {
            fov_encoder,
            linear: LinearConfig::new(EMBED_DIM, num_features / 2).init(device),
        };

        FOVNetwork {
            encoder,
            downsample: vec![fov_head0],
            head: fov_head,
        }
    }
}
