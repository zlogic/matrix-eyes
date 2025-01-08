use candle_core::{Result, Tensor};
use candle_nn::{
    Activation, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder,
};

struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    fn new(num_features: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        let vb = vb.pp("residual");
        let conv1 = candle_nn::conv2d(num_features, num_features, 3, conv_cfg, vb.pp(1))?;
        let conv2 = candle_nn::conv2d(num_features, num_features, 3, conv_cfg, vb.pp(3))?;

        Ok(Self { conv1, conv2 })
    }
}

impl Module for ResidualConvUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let activation = Activation::Relu;

        println!("conv1");
        let out = activation.forward(xs)?;
        let out = self.conv1.forward(&out)?;

        println!("conv2");
        let out = activation.forward(&out)?;
        let out = self.conv2.forward(&out)?;

        println!("out");
        out + xs
    }
}

struct FeatureFusionBlock {
    res_conv_unit1: ResidualConvUnit,
    res_conv_unit2: ResidualConvUnit,
    deconv: Option<ConvTranspose2d>,
    output_conv: Conv2d,
}

impl FeatureFusionBlock {
    fn new(num_features: usize, deconv: bool, vb: VarBuilder) -> Result<Self> {
        let res_conv_unit1 = ResidualConvUnit::new(num_features, vb.pp("resnet1"))?;
        let res_conv_unit2 = ResidualConvUnit::new(num_features, vb.pp("resnet2"))?;

        let deconv = if deconv {
            Some(candle_nn::conv_transpose2d_no_bias(
                num_features,
                num_features,
                2,
                ConvTranspose2dConfig {
                    padding: 0,
                    output_padding: 0,
                    stride: 2,
                    dilation: 1,
                },
                vb.pp("deconv"),
            )?)
        } else {
            None
        };

        let output_conv = candle_nn::conv2d(
            num_features,
            num_features,
            1,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("out_conv"),
        )?;

        Ok(Self {
            res_conv_unit1,
            res_conv_unit2,
            deconv,
            output_conv,
        })
    }

    fn forward(&self, x0: Tensor, mut x1: Option<Tensor>) -> Result<Tensor> {
        let out = if let Some(x1) = x1.take() {
            // skip_add in PyTorch is just a regular addition.
            let res = self.res_conv_unit1.forward(&x1)?;
            println!("resnet1 {:?} {:?}", x0.dims(), res.dims());
            (x0 + res)?
        } else {
            x0
        };

        println!("resnet2");
        let mut out = self.res_conv_unit2.forward(&out)?;

        if let Some(ref deconv) = self.deconv {
            out = deconv.forward(&out)?;
        }

        self.output_conv.forward(&out)
    }
}

pub(super) struct MultiresConvDecoder {
    convs: Vec<Conv2d>,
    fusions: Vec<FeatureFusionBlock>,
}

impl MultiresConvDecoder {
    pub fn new(
        vb: VarBuilder,
        dims_encoder: &[usize],
        dim_decoder: usize,
    ) -> Result<MultiresConvDecoder> {
        let mut convs = vec![];
        if dims_encoder[0] != dim_decoder {
            println!("Add conv0");
            convs.push(candle_nn::conv2d_no_bias(
                dims_encoder[0],
                dim_decoder,
                1,
                Conv2dConfig::default(),
                vb.pp("convs").pp(0),
            )?)
        };
        for (i, dims_encoder_i) in dims_encoder.iter().enumerate().skip(1) {
            convs.push(candle_nn::conv2d_no_bias(
                *dims_encoder_i,
                dim_decoder,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("convs").pp(i),
            )?)
        }

        let fusions = (0..dims_encoder.len())
            .map(|i| FeatureFusionBlock::new(dim_decoder, i != 0, vb.pp("fusions").pp(i)))
            .collect::<Result<Vec<_>>>()?;

        Ok(MultiresConvDecoder { convs, fusions })
    }

    pub fn forward(&self, encodings: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        if encodings.len() != self.fusions.len() {
            let received = encodings.len();
            let expected = self.fusions.len();
            candle_core::bail!("got encoder output levels {received}, expected levels {expected}")
        }

        let lowres_features = self
            .convs
            .last()
            .unwrap()
            .forward(encodings.last().unwrap())?;
        println!("lowres features {:?}", lowres_features.dims());
        let mut features = self
            .fusions
            .last()
            .unwrap()
            .forward(lowres_features.clone(), None)?;
        println!("lowres features {:?}", features.dims());

        for (i, encoding) in encodings.into_iter().enumerate().rev().skip(1) {
            println!(
                "fusionconvs {} {} {}",
                i,
                self.convs.len(),
                self.fusions.len()
            );
            let conv = if self.convs.len() == self.fusions.len() {
                Some(&self.convs[i])
            } else if i >= 1 {
                Some(&self.convs[i - 1])
            } else {
                None
            };
            let features_i = if let Some(conv) = conv {
                conv.forward(&encoding)?
            } else {
                encoding
            };
            println!(
                "features {:?}  features_i {:?}",
                features.dims(),
                features_i.dims()
            );
            features = self.fusions[i].forward(features, Some(features_i))?;
        }

        Ok((features, lowres_features))
    }
}
