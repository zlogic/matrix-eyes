use candle_core::{IndexOp as _, Tensor};

pub fn debug_tensor(x: &Tensor) -> Result<(), candle_core::Error> {
    if x.dims().len() == 2 {
        for i in 0..x.dim(0)? {
            let x = x.i(i)?;
            println!("{:?}", x);
        }
    } else if x.dims().len() == 1 {
        println!("{:?}", x);
    } else {
        println!("??? {:?}", x.dims());
    }
    Ok(())
}
