use candle_core::Tensor;
use candle_nn::{Activation, Conv2dConfig, ConvTranspose2dConfig, Module as _, VarBuilder};
use decoder::MultiresConvDecoder;
use encoder::DepthProEncoder;

mod decoder;
mod encoder;

pub fn extract_depth(
    vb: VarBuilder,
    img: &Tensor,
    f_norm: f32,
) -> Result<Tensor, candle_core::Error> {
    const ENCODER_FEATURE_DIMS: [usize; 4] = [256, 512, 1024, 1024];
    const DECODER_FEATURES: usize = 256;

    let encodings = {
        let encoder =
            DepthProEncoder::new(vb.pp("encoder"), &ENCODER_FEATURE_DIMS, DECODER_FEATURES)?;
        encoder.forward_encodings(img)?
    };

    let (mut features, features_0) = {
        let mut dims_encoder = vec![DECODER_FEATURES];
        dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
        let decoder = MultiresConvDecoder::new(vb.pp("decoder"), &dims_encoder, DECODER_FEATURES)?;
        decoder.forward(encodings)?
    };
    drop(features_0);

    let canonical_inverse_depth = {
        println!("Head");
        let head_layer = candle_nn::conv2d(
            DECODER_FEATURES,
            DECODER_FEATURES / 2,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("head").pp(0),
        )?;
        features = head_layer.forward(&features)?;

        let head_layer = candle_nn::conv_transpose2d(
            DECODER_FEATURES / 2,
            DECODER_FEATURES / 2,
            2,
            ConvTranspose2dConfig {
                padding: 0,
                output_padding: 0,
                stride: 2,
                dilation: 1,
            },
            vb.pp("head").pp(1),
        )?;
        features = head_layer.forward(&features)?;

        let head_layer = candle_nn::conv2d(
            DECODER_FEATURES / 2,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            vb.pp("head").pp(2),
        )?;
        features = head_layer.forward(&features)?;

        let head_layer = Activation::Relu;
        features = head_layer.forward(&features)?;

        let head_layer = candle_nn::conv2d(32, 1, 1, Conv2dConfig::default(), vb.pp("head").pp(4))?;
        features = head_layer.forward(&features)?;

        let head_layer = Activation::Relu;
        head_layer.forward(&features)?
    };
    drop(features);
    println!("head done");
    let canonical_inverse_depth = canonical_inverse_depth.squeeze(0)?.squeeze(0)?;

    let inverse_depth =
        canonical_inverse_depth.broadcast_div(&Tensor::new(f_norm, img.device())?)?;

    let depth = (1.0 / inverse_depth.clamp(1e-4, 1e4)?)?;

    Ok(depth)
}
