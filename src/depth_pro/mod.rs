use candle_nn::VarBuilder;

mod dino_vit;

pub struct Encoder {}

impl Encoder {
    pub fn new(vb: VarBuilder) -> Result<Encoder, candle_core::Error> {
        //let vb = vb.pp("encoder").pp("image_encoder");
        let vb = vb.pp("encoder").pp("patch_encoder");
        dino_vit::dinov2l16_384(vb)?;
        Ok(Encoder {})
    }
}
