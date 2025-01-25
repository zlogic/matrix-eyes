use std::{env, process::exit};

mod depth_pro;
mod output;
mod reconstruction;

#[derive(Debug)]
pub struct Args {
    focal_length: Option<f32>,
    checkpoint_path: String,
    convert_checkpoints: bool,
    output_format: output::ImageOutputFormat,
    img_src: String,
    img_out: String,
}

const USAGE_INSTRUCTIONS: &str = "Usage: matrix-eyes [OPTIONS] <IMG_SRC>... <IMG_OUT>\n\n\
Arguments:\
\n  <IMG_SRC>...  Source image\
\n  <IMG_OUT>     Output image\n\n\
Options:\
\n      --focal-length=<FOCAL_LENGTH>       Focal length in 35mm equivalent\
\n      --checkpoint-path=<CHECKPOINT_PATH> Path to checkpoint file [default: ./checkpoints/depth_pro.pt]\
\n      --output-format=<FORMAT>            Format for output [default: depthmap for images, mesh for meshes] [possible values: depthmap, stereogram]\
\n      --resize-scale=<SCALE>              Custom scale for stereogram output [default: 1.0]\
\n      --stereo-amplitude=<AMPLITUDE>      Custom scale for stereogram output [default: 0.0625]\
\n      --convert-checkpoints               Convert checkpoints into a more efficient format [default: disabled]\
\n      --help                              Print help";

impl Args {
    fn parse() -> Args {
        let mut args = Args {
            focal_length: None,
            checkpoint_path: "./checkpoints/depth_pro.pt".to_string(),
            convert_checkpoints: false,
            output_format: output::ImageOutputFormat::DepthMap,
            img_src: "".to_string(),
            img_out: "".to_string(),
        };
        let mut resize_scale = None;
        let mut stereo_amplitude = 1.0 / 16.0;
        let default_stereogram =
            output::ImageOutputFormat::Stereogram(resize_scale, stereo_amplitude);
        for arg in env::args().skip(1) {
            if arg.starts_with("--") && args.img_src.is_empty() && args.img_out.is_empty() {
                // Option flags.
                if arg == "--convert-checkpoints" {
                    args.convert_checkpoints = true;
                    continue;
                } else if arg == "--help" {
                    println!("{}", USAGE_INSTRUCTIONS);
                    exit(0);
                }
                let (name, value) = if let Some(arg) = arg.split_once('=') {
                    arg
                } else {
                    eprintln!("Option flag {} has no value", arg);
                    println!("{}", USAGE_INSTRUCTIONS);
                    exit(2);
                };
                if name == "--focal-length" {
                    args.focal_length = match value.parse() {
                        Ok(focal_length) => Some(focal_length),
                        Err(err) => {
                            eprintln!(
                                "Argument {} has an unsupported value {}: {}",
                                name, value, err
                            );
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2)
                        }
                    };
                } else if name == "--output-format" {
                    args.output_format = match value.to_lowercase().as_str() {
                        "depthmap" => output::ImageOutputFormat::DepthMap,
                        "stereogram" => default_stereogram.clone(),
                        _ => {
                            eprintln!("Unsupported output format {}", value);
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2)
                        }
                    };
                } else if name == "--resize-scale" {
                    resize_scale = match value.parse() {
                        Ok(resize_scale) => Some(resize_scale),
                        Err(err) => {
                            eprintln!(
                                "Argument {} has an unsupported value {}: {}",
                                name, value, err
                            );
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2)
                        }
                    };
                } else if name == "--stereo-amplitude" {
                    stereo_amplitude = match value.parse() {
                        Ok(stereo_amplitude) => stereo_amplitude,
                        Err(err) => {
                            eprintln!(
                                "Argument {} has an unsupported value {}: {}",
                                name, value, err
                            );
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2)
                        }
                    };
                } else if name == "--checkpoint-path" {
                    args.checkpoint_path = value.to_string();
                } else {
                    eprintln!("Unsupported argument {}", arg);
                }
            } else if args.img_src.is_empty() {
                args.img_src = arg;
            } else if args.img_out.is_empty() {
                args.img_out = arg
            } else {
                eprintln!("Unexpected argument {}", arg);
                println!("{}", USAGE_INSTRUCTIONS);
                exit(2)
            }
        }

        match args.output_format {
            output::ImageOutputFormat::Stereogram(_, _) => {
                args.output_format =
                    output::ImageOutputFormat::Stereogram(resize_scale, stereo_amplitude)
            }
            _ => {}
        }

        if args.img_src.is_empty() {
            eprintln!("No source image provided");
            println!("{}", USAGE_INSTRUCTIONS);
            exit(2);
        } else if args.img_out.is_empty() {
            eprintln!("No output image provided");
            println!("{}", USAGE_INSTRUCTIONS);
            exit(2);
        }

        args
    }
}

fn main() {
    println!(
        "Matrix Eyes version {}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown")
    );

    let args = Args::parse();

    let device = reconstruction::init_device();
    let model = match depth_pro::DepthProModel::<reconstruction::EnabledBackend>::new(
        &args.checkpoint_path,
        args.convert_checkpoints,
        &device,
    ) {
        Ok(model) => model,
        Err(err) => {
            println!("Failed to create Depth Pro model: {}", err);
            exit(1);
        }
    };

    if let Err(err) = reconstruction::extract_depth(
        &device,
        &model,
        &args.img_src,
        &args.img_out,
        args.focal_length,
        args.output_format,
    ) {
        println!("Reconstruction failed: {}", err);
        exit(1);
    }
}
