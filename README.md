# Matrix Eyes

![Build status](https://github.com/zlogic/matrix-eyes/actions/workflows/cargo-build.yml/badge.svg)

Matrix Eyes is a weekend project to convert a photo image into an [autostereogram](https://en.wikipedia.org/wiki/Autostereogram).

Using the [MiDaS](https://arxiv.org/abs/1907.01341) and [Apple Depth Pro](https://arxiv.org/abs/2410.02073) depth estimation algorithms.

# Instructions

## Installation

```shell
pip install -r requirements.txt
```

For MiDaS, install `opencv-python` as well:

```shell
pip install opencv-python
```

For Depth Pro, download the model checkpoints:

```shell
mkdir checkpoints
curl -LJ -o checkpoints/depth_pro.pt https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
```

## Usage

To use a custom Torch home directory, set the `TORCH_HOME` environment variable.

To gerate a depth image, run:

```shell
python main.py [--model-type=Depth_Pro|DPT_Large|DPT_Hybrid|MiDaS_small] [--output-format=image|stereogram|mesh] [--stereo-amplitude=<value>] <input file> <output_file>
```

replacing `<input file>` with the source image filename, and `<output file>` with the output destination filename.

Additional (optional) arguments:

* `model-type` specifies one of the MiDaS models
    * `Depth_Pro` will use Depth Pro (alternative to MiDaS)
    * `DPT_Large` will use MiDaS v3 - Large (highest accuracy, slowest inference speed); default
    * `DPT_Hybrid` will use MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
    * `MiDaS_small` will use MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
* `output-format` specifies what to output
    * `image` will output a depth map image; default
    * `stereogram` will output a stereogram image
    * `mesh` will output a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
* `stereo-amplitude` specifies the maximum offset/depth for stereograms (relative to image width); might need to be reduced if most of the image consists of foreground objects

# Examples

## Image 1

Source image:

![Source image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.jpg)

Depth data extracted by MiDaS:

![Depth data for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.depth.jpg)

Generated stereogram - works best when [viewed in fullscreen](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereo.jpg):

![Stereogram for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereo.jpg)
