use std::{env, process::exit};

mod reconstruction;

#[derive(Debug)]
pub struct Args {
    focal_length: Option<f32>,
    img_src: String,
    img_out: String,
}

const USAGE_INSTRUCTIONS: &str = "Usage: matrix-eyes [OPTIONS] <IMG_SRC>... <IMG_OUT>\n\n\
Arguments:\
\n  <IMG_SRC>...  Source image\
\n  <IMG_OUT>     Output image\n\n\
Options:\
\n      --focal-length=<FOCAL_LENGTH>    Focal length in 35mm equivalent\
\n      --help                           Print help";

impl Args {
    fn parse() -> Args {
        let mut args = Args {
            focal_length: None,
            img_src: "".to_string(),
            img_out: "".to_string(),
        };
        for arg in env::args().skip(1) {
            if arg.starts_with("--") && args.img_src.is_empty() && args.img_out.is_empty() {
                // Option flags.
                if arg == "--help" {
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

    if let Err(err) = reconstruction::extract_depth(&args.img_src, args.focal_length) {
        println!("Reconstruction failed: {}", err);
        exit(1);
    }
}