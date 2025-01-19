use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        PaddingConfig2d, Relu,
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
struct ResidualConvUnit<B: Backend> {
    residual: Vec<Conv2d<B>>,
}

impl<B> ResidualConvUnit<B>
where
    B: Backend,
{
    fn new(device: &B::Device, num_features: usize) -> ResidualConvUnit<B> {
        let conv1 = Conv2dConfig::new([num_features, num_features], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let conv2 = Conv2dConfig::new([num_features, num_features], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let residual = vec![conv1, conv2];

        ResidualConvUnit { residual }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let activation = Relu::new();
        let mut out = input.clone();
        for conv in &self.residual {
            out = activation.forward(out);
            out = conv.forward(out);
        }

        input + out
    }
}

#[derive(Module, Debug)]
struct FeatureFusionBlock<B: Backend> {
    resnet1: ResidualConvUnit<B>,
    resnet2: ResidualConvUnit<B>,
    deconv: Option<ConvTranspose2d<B>>,
    out_conv: Conv2d<B>,
}

impl<B> FeatureFusionBlock<B>
where
    B: Backend,
{
    fn new(device: &B::Device, num_features: usize, deconv: bool) -> FeatureFusionBlock<B> {
        let resnet1 = ResidualConvUnit::new(device, num_features);
        let resnet2 = ResidualConvUnit::new(device, num_features);

        let deconv = if deconv {
            Some(
                ConvTranspose2dConfig::new([num_features, num_features], [2, 2])
                    .with_stride([2, 2])
                    .with_bias(false)
                    .init(device),
            )
        } else {
            None
        };

        let out_conv = Conv2dConfig::new([num_features, num_features], [1, 1]).init(device);

        FeatureFusionBlock {
            resnet1,
            resnet2,
            deconv,
            out_conv,
        }
    }

    fn forward(&self, x0: Tensor<B, 4>, mut x1: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let out = if let Some(x1) = x1.take() {
            // skip_add in PyTorch is just a regular addition.
            let res = self.resnet1.forward(x1);
            x0 + res
        } else {
            x0
        };

        let out = self.resnet2.forward(out);

        let out = if let Some(ref deconv) = self.deconv {
            deconv.forward(out)
        } else {
            out
        };

        self.out_conv.forward(out)
    }
}

#[derive(Module, Debug)]
pub(super) struct MultiresConvDecoder<B: Backend> {
    convs: Vec<Conv2d<B>>,
    fusions: Vec<FeatureFusionBlock<B>>,
}

#[derive(Config, Debug)]
pub(super) struct MultiresConvDecoderConfig {}

impl MultiresConvDecoderConfig {
    pub fn init<B>(
        device: &B::Device,
        dims_encoder: &[usize],
        dim_decoder: usize,
    ) -> MultiresConvDecoder<B>
    where
        B: Backend,
    {
        let mut convs = if dims_encoder[0] != dim_decoder {
            vec![Conv2dConfig::new([dims_encoder[0], dim_decoder], [1, 1])
                .with_bias(false)
                .init(device)]
        } else {
            vec![]
        };
        for dims_encoder_i in dims_encoder.iter().skip(1) {
            convs.push(
                Conv2dConfig::new([*dims_encoder_i, dim_decoder], [3, 3])
                    .with_bias(false)
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        }

        let fusions = (0..dims_encoder.len())
            .map(|i| FeatureFusionBlock::new(device, dim_decoder, i != 0))
            .collect::<Vec<_>>();

        MultiresConvDecoder { convs, fusions }
    }
}

impl<B> MultiresConvDecoder<B>
where
    B: Backend,
{
    pub fn forward(&self, mut encodings: Vec<Tensor<B, 4>>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if encodings.len() != self.fusions.len() {
            let received = encodings.len();
            let expected = self.fusions.len();
            panic!("got encoder output levels {received}, expected levels {expected}")
        }

        let last_encoding = encodings.pop().expect("empty encodings list");
        let mut features = self
            .convs
            .last()
            .expect("empty convs block list")
            .forward(last_encoding);
        let lowres_features = features.clone();
        features = self
            .fusions
            .last()
            .expect("empty fusions block list")
            .forward(features, None);

        for (i, encoding) in encodings.into_iter().enumerate().rev() {
            let conv = if self.convs.len() == self.fusions.len() {
                Some(&self.convs[i])
            } else if i >= 1 {
                Some(&self.convs[i - 1])
            } else {
                None
            };
            let features_i = if let Some(conv) = conv {
                conv.forward(encoding)
            } else {
                encoding
            };
            features = self.fusions[i].forward(features, Some(features_i));
        }

        (features, lowres_features)
    }
}
