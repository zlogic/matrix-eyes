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
        let encodings = self.encoder.forward_encodings(img)?;
        println!(
            "Extracted {:?} {:?} {:?} {:?} {:?}",
            encodings[0].dims(),
            encodings[1].dims(),
            encodings[2].dims(),
            encodings[3].dims(),
            encodings[4].dims()
        );
        Ok(())
    }
}
