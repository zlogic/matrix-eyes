use candle_nn::VarBuilder;
use decoder::MultiresConvDecoder;
use encoder::DepthProEncoder;

mod decoder;
mod encoder;

pub struct DepthPro {
    encoder: DepthProEncoder,
    decoder: MultiresConvDecoder,
}

impl DepthPro {
    pub fn new(vb: VarBuilder) -> Result<DepthPro, candle_core::Error> {
        const ENCODER_FEATURE_DIMS: [usize; 4] = [256, 512, 1024, 1024];
        const DECODER_FEATURES: usize = 256;

        let encoder =
            DepthProEncoder::new(vb.pp("encoder"), &ENCODER_FEATURE_DIMS, DECODER_FEATURES)?;

        let mut dims_encoder = vec![DECODER_FEATURES];
        dims_encoder.extend_from_slice(&ENCODER_FEATURE_DIMS);
        let decoder = MultiresConvDecoder::new(vb.pp("decoder"), &dims_encoder, DECODER_FEATURES)?;

        Ok(DepthPro { encoder, decoder })
    }

    pub fn extract_depth(&self, img: &candle_core::Tensor) -> Result<(), candle_core::Error> {
        let encodings = self.encoder.forward_encodings(img)?;
        let (features, features_0) = self.decoder.forward(encodings)?;
        Ok(())
    }
}
