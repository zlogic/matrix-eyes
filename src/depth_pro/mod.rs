use candle_nn::{Module as _, VarBuilder};
use vit::DepthProEncoder;

mod vit;

pub struct DepthPro {
    encoder: DepthProEncoder,
}

impl DepthPro {
    pub fn new(vb: VarBuilder) -> Result<DepthPro, candle_core::Error> {
        let encoder = DepthProEncoder::new(vb.pp("encoder"))?;

        Ok(DepthPro { encoder })
    }

    pub fn extract_depth(&self, img: &candle_core::Tensor) -> Result<(), candle_core::Error> {
        self.encoder.forward(img)?;
        Ok(())
    }
}
