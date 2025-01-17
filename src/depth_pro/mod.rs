use candle_core::{IndexOp as _, Tensor};
use candle_nn::{Activation, Conv2dConfig, ConvTranspose2dConfig, Module as _, VarBuilder};
use decoder::MultiresConvDecoder;
use encoder::DepthProEncoder;

use crate::de;

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

    de::debug_tensor(&encodings[0].i((0, 100, 24..28, 24..28))?)?;
    de::debug_tensor(&encodings[1].i((0, 100, 24..28, 24..28))?)?;
    println!("");
    //de::debug_tensor(&encodings[2].i((0, 100, 24..28, 24..28))?)?;
    //de::debug_tensor(&encodings[3].i((0, 100, 24..28, 24..28))?)?;
    //de::debug_tensor(&encodings[4].i((0, 100, 24..28, 24..28))?)?;

    let (features, features_0) = {
        let mut dims_encoder = vec![DECODER_FEATURES];
        dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
        let decoder = MultiresConvDecoder::new(vb.pp("decoder"), &dims_encoder, DECODER_FEATURES)?;
        decoder.forward(encodings)?
    };
    drop(features_0);

    let canonical_inverse_depth = {
        let mut head = candle_nn::seq();
        head = head.add(candle_nn::conv2d(
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
        )?);
        head = head.add(candle_nn::conv_transpose2d(
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
        )?);
        head = head.add(candle_nn::conv2d(
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
        )?);
        head = head.add(Activation::Relu);
        head = head.add(candle_nn::conv2d(
            32,
            1,
            1,
            Conv2dConfig::default(),
            vb.pp("head").pp(4),
        )?);
        head = head.add(Activation::Relu);

        println!("features before {:?}", features.dims());
        de::debug_tensor(&features.i((0, 0, 20..24, 100..105))?)?;

        head.forward(&features)?
    };
    println!("features after {:?}", features.dims());
    de::debug_tensor(&canonical_inverse_depth.i((0, 0, 20..24, 100..105))?)?;
    let canonical_inverse_depth = canonical_inverse_depth.squeeze(0)?.squeeze(0)?;

    /*
    let inverse_depth =
        canonical_inverse_depth.broadcast_div(&Tensor::new(f_norm, img.device())?)?;

    let depth = (1.0 / inverse_depth.clamp(1e-4, 1e4)?)?;

    Ok(depth)
    */

    Ok(canonical_inverse_depth)
}
