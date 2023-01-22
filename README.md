# Matrix Eyes

Matrix Eyes is a weekend project to convert a photo image into an [autostereogram](https://en.wikipedia.org/wiki/Autostereogram).

Using the [MiDaS](https://arxiv.org/abs/1907.01341) depth estimation algorithm.

# Instructions

## Installation

```shell
pip install -r requirements.txt
```

## Usage

To use a custom Torch home directory, set the `TORCH_HOME` environment variable.

To gerate a depth image, run:

```shell
python main.py [--model-type=DPT_Large|DPT_Hybrid|MiDaS_small] [--output-format=image|stereogram|mesh] [--stereo-amplitude=<value>] <input file> <output_file>
```

replacing `<input file>` with the source image filename, and `<output file>` with the output destination filename.

Additional (optional) arguments:

* `model-type` specifies one of the MiDaS models
    * `DPT_Large` will use MiDaS v3 - Large (highest accuracy, slowest inference speed); default
    * `DPT_Hybrid` will use MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
    * `MiDaS_small` will use MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
* `output-format` specifies what to output
    * `image` will output a depth map image; default
    * `stereogram` will output a stereogram image
    * `mesh` will output a 3D [Wavefront OBJ file](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
* `stereo-amplitude` specifies the maximuim offset/depth for stereograms (relative to image width); might need to be reduced if most of the image consists of foreground objects

# Examples

## Image 1

Source image:

![Source image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.jpg)

Depth data extracted by MiDaS:

![Depth data for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.depth.jpg)

Generated stereogram - works best when [viewed in fullscreen](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereo.jpg):

![Stereogram for image 1](https://raw.githubusercontent.com/wiki/zlogic/matrix-eyes/Examples/img1.stereo.jpg)
