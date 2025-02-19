# Matrix Eyes

![Build status](https://github.com/zlogic/matrix-eyes/actions/workflows/cargo-build.yml/badge.svg)

Matrix Eyes is a Rust port of [Apple Depth Pro](https://github.com/apple/ml-depth-pro) project to convert a photo image into an [autostereogram](https://en.wikipedia.org/wiki/Autostereogram) or 3D mesh.

For running ML models, the [burn](https://github.com/tracel-ai/burn) library is used. There's also an experimental version using [candle](https://github.com/huggingface/candle) in the [candle tag](https://github.com/zlogic/matrix-eyes/tree/candle).

The [python tag](https://github.com/zlogic/matrix-eyes/tree/python) contains an older Python-based version which supported both [MiDaS](https://arxiv.org/abs/1907.01341) and [Apple Depth Pro](https://arxiv.org/abs/2410.02073) depth estimation algorithms.

This app reuses some code from [Cybervision](https://github.com/zlogic/cybervision), and tries to achieve the same goal. Cybervision uses a "classic" structure-from-motion approach and reconstructs objects from multiple views, while Matrix Eyes uses a pretrained machine learning model to add depth to a single image.

# Examples

## Image 1

Source image:

![Source image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.jpg)

Depth data extracted by Matrix Eyes:

![Depth data for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.depth.png)

Generated stereogram - works best when [viewed in fullscreen](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereo.jpg):

![Stereogram for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereogram.png)

![Example mesh](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/photo5.mov)

# Instructions

## Installation

Download a copy of Matrix Eyes from [Releases](releases) and extract it into in a directory.
The following versions are available:

* Windows
  * ndarray (slow, CPU-only version)
  * candle-cuda (fastest version, using the candle backend and cuDNN libraries)
  * wgpu-spirv (vendor-neutral GPU version, fails to run on a GPU with 8GB of VRAM)
* Ubuntu
  * ndarray (slow, CPU-only version)
  * candle-cuda (not tested)
  * wgpu-spirv (not tested)
* macOS
  * wgpu-spirv (wgpu version, uses 12+ GB when running)
  * ndarray-accelerate (slow, CPU-only version that might be using AMX instructions)

For the Windows CUDA version, download the [CUDA libraries](https://github.com/zlogic/matrix-eyes/releases/download/0.1.0/cuda-Windows-x86_64.zip) artifact and extract its contents into the same directory.

Download the model checkpoints:

```shell
mkdir checkpoints
curl -LJ -o checkpoints/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
```

## Usage

To gerate a depth image, run:

```shell
matrix-eyes [--focal-length=<focal-length>] [--checkpoint-path=<checkpoint-path>] [--image-output-format=<depthmap|stereogram>] [--resize-scale=<scale>] [--stereo-amplitude=<amplitude>] [--mesh=<plain|vertex-colors|texture-coordinates>] [--convert-checkpoints] <source> <output>
```

`--focal-length=<focal-length>` is an optional argument to specify a custom focal length for images with perspective projection, for example, `--focal-length=26`;
this should be the image's focal length in 35mm equivalent.
If not specified, EXIF metadata will be used; if EXIF data is not available, the focal length will be estimated using Depth Pro.

`--checkpoint-path=<checkpoint-path>` is an optional argument to specify a custom path to the Depth Pro checkpoints file, `--checkpoint-path=./ckpoint.pt`.

`--image-output-format=<depthmap|stereogram>` is an optional argument to specify the image output format, for example `--image-output-format=depthmap` or `--image-output-format=stereogram`.
`depthmap` (the default option) outputs a depth map image, while `--image-output-format=stereogram` outputs a stereogram image.

`--resize-scale=<scale>` is an optional argument to specify a custom scale for the stereogram image output, for example `--resize-scale=0.25`.
This can help with making noise pixels large enough to be visible.

`--stereo-amplitude` is an optional argument to specify the maximum offset/depth for stereograms (relative to image width); might need to be reduced if most of the image consists of foreground objects, for example `--stereo-amplitude=0.0625`.

`--mesh=<plain|vertex-colors|texture-coordinates>` is an optional argument to specify how to output OBJ and PLY meshes mode, for example `--mesh=vertex-colors` or `--mesh=texture-coordinates`.
`plain` (the default option) outputs the mesh without any color or texture, `vertex-colors` outputs the mesh with colors assigned to every vertex, and `texture-coordinates` will add texture coordinates.

`--convert-checkpoints` will convert checkpoints from a `.pt` (Python pickle) format into a more efficient Burn format.

`<source>` specifies the filename for the source file; supported formats are `jpg` and `png`.

`<output>` is the output filename:
* If the filename ends with `.obj`, this will save a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file).
* If the filename ends with `.ply`, this will save a 3D [PLY binary file](https://en.wikipedia.org/wiki/PLY_(file_format)).
* If the filename ends with `.png`, this will save a PNG image (depth map or stereogram).
* If the filename ends with `.jpg`, this will save a JPEG image (depth map or stereogram).

### GPU details

Matrix Eyes was tested to support CPU-only and GPU-accelerated processing on:

* Apple Macbook Pro M1 Max (2021) (ndarray-accelerate, ndarray and wgpu versions)
* Windows 11, i7-11800H, Geforce RTX 3070 (candle-cuda and ndarray versions)
* Fedora 41 in WSL (ndarray version)

