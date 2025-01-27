use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, PaddingConfig2d, Relu,
    },
    prelude::Backend,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Tensor,
    },
};

use super::vit::{self, DinoVisionTransformer};

const EMBED_DIM: usize = vit::EMBED_DIM;

#[derive(Module, Debug)]
enum EncoderBlock<B: Backend> {
    Encoder(DinoVisionTransformer<B>),
    Linear(Linear<B>),
}

#[derive(Module, Debug)]
pub(super) struct FOVNetwork<B: Backend> {
    encoder: Vec<EncoderBlock<B>>,
    downsample: Vec<Conv2d<B>>,
    head: Vec<Conv2d<B>>,
}

impl<B> FOVNetwork<B>
where
    B: Backend,
{
    pub fn forward(&self, x: Tensor<B, 4>, lowres_feature: Tensor<B, 4>) -> Tensor<B, 1> {
        const INTERPOLATE_MODE: InterpolateMode = if cfg!(feature = "candle-cuda") {
            InterpolateMode::Nearest
        } else {
            InterpolateMode::Bilinear
        };
        let [_b, _c, h, w] = x.dims();
        println!("interpolate");
        let x = interpolate(x, [w / 4, h / 4], InterpolateOptions::new(INTERPOLATE_MODE));
        let x = if let [EncoderBlock::Encoder(encoder), EncoderBlock::Linear(linear)] =
            self.encoder.as_slice()
        {
            println!("encoder forward");
            let x = encoder.forward_features(x, &[]).0;
            println!("linear forward");
            linear.forward(x)
        } else {
            panic!("fov encoder has unexpected blocks configuration")
        };
        let [_, x_dim1, _] = x.dims();
        println!("x narrow {:?}", x.dims());
        let x = x.narrow(1, 1, x_dim1 - 1).permute([0, 2, 1]);

        println!("downsample forward {:?}", lowres_feature.dims());
        let lowres_feature = self.downsample[0].forward(lowres_feature);
        let lowres_feature = Relu::new().forward(lowres_feature);
        println!("lowres add {:?} {:?}", lowres_feature.dims(), x.dims());
        let x = x.reshape(lowres_feature.dims()) + lowres_feature;

        println!("x head0 {:?}", x.dims());
        let x = self.head[0].forward(x);
        println!("x relu {:?}", x.dims());
        let x = Relu::new().forward(x);
        println!("x head1 {:?}", x.dims());
        let x = self.head[1].forward(x);
        println!("x relu {:?}", x.dims());
        let x = Relu::new().forward(x);
        println!("x head2 {:?}", x.dims());
        let x = self.head[2].forward(x);

        println!("x squeeze {:?}", x.dims());
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

        let encoder = vec![
            EncoderBlock::Encoder(fov_encoder),
            EncoderBlock::Linear(LinearConfig::new(EMBED_DIM, num_features / 2).init(device)),
        ];

        FOVNetwork {
            encoder,
            downsample: vec![fov_head0],
            head: fov_head,
        }
    }
}
