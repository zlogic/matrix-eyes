import argparse
import numpy as np
from cmap import Colormap
from PIL import Image
import models

def output_mesh(depth, filename):
    width = depth.shape[1]
    height = depth.shape[0]

    depth_min = depth.min()
    depth_max = depth.max()

    depth = min(width, height) * (depth - depth_min) / (depth_max - depth_min)
    with open(filename, 'w') as f:
        for y in range(height):
            for x in range(width):
                z = depth[y][x]
                f.write(f'v {x} {height-y} {z}\n')

        for y in range(height-1):
            for x in range(width-1):
                f.write(f'f {y*width+x+1} {(y+1)*width+x+1} {y*width+(x+1)+1}\n')
                f.write(f'f {y*width+(x+1)+1} {(y+1)*width+x+1} {(y+1)*width+(x+1)+1}\n')

def output_image(depth, filename):
    depth_min = depth.min()
    depth_max = depth.max()

    out = (depth - depth_min) / (depth_max - depth_min)
    cmap = Colormap('inferno')
    out = (cmap(out)[..., :3] * 255.0).astype(np.uint8)

    Image.fromarray(out).save(filename)

def output_stereogram(depth, filename, amplitude):
    width = depth.shape[1]
    height = depth.shape[0]
    out = np.zeros((height, width, 3), np.uint8)

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    depth_multiplier = width*amplitude
    pattern_width = int(depth_multiplier*2+16)
    pattern = np.random.randint(0, 255, (height, pattern_width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            if x >= pattern_width:
                shift = int(depth[y][x] * depth_multiplier)
                out[y][x] = out[y][x - pattern_width + shift]
            else:
                out[y][x] = pattern[y][x%pattern_width]

    Image.fromarray(out).save(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',
        choices=['Depth_Pro', 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
        default='DPT_Large')
    parser.add_argument('--output-format', choices=['image', 'mesh', 'stereogram'], default='image')
    parser.add_argument('--resize-scale', type=float, default=1.0)
    parser.add_argument('--stereo-amplitude', type=float, default=1.0/16)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    if args.model_type == 'Depth_Pro':
        model = models.DepthPro()
    else:
        model = models.MiDaS(args.model_type)

    output = model.extract_depth(args.input, args.resize_scale)

    del(model)
    if args.output_format == 'image':
        output_image(output, args.output)
    elif args.output_format == 'mesh':
        output_mesh(output, args.output)
    elif args.output_format == 'stereogram':
        output_stereogram(output, args.output, args.stereo_amplitude)

if __name__ == '__main__':
    main()
