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
python main.py [--model-type=DPT_Large|DPT_Hybrid|MiDaS_small] <input file> <output_file>
```

replacing `<input file>` with the source image filename, and `<output file>` with the output destination filename.

Additional (optional) arguments:

* `model-type` specifies one of the MiDaS models
    * `DPT_Large` will use MiDaS v3 - Large (highest accuracy, slowest inference speed)
    * `DPT_Hybrid` will use MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
    * `MiDaS_small` will use MiDaS v2.1 - Small (lowest accuracy, highest inference speed)
