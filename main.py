import argparse
import numpy as np
from cmap import Colormap
from PIL import Image
import models

def output_mesh(depth, image, f_px, filename):
    width = depth.shape[1]
    height = depth.shape[0]

    depth_min = depth.min()
    depth_max = depth.max()

    with open(filename, 'w') as f:
        for y in range(height):
            for x in range(width):
                z = depth[y][x]
                color = image[y][x]/255.0
                x_out, y_out = x-(width/2), height/2-y
                if f_px is not None:
                    # Clip depth to avoid points at infinity
                    z = max(-250.0, -z)
                    x_out = -x_out*z/f_px
                    y_out = -y_out*z/f_px
                else:
                    # Use a reasonable depth scale
                    z = max(width, height) * (z-depth_min) / (depth_max - depth_min)
                f.write(f'v {x_out} {y_out} {z} {color[0]} {color[1]} {color[2]}\n')

        for y in range(height-1):
            for x in range(width-1):
                d_min, d_max = depth[y][x], depth[y][x]
                for j in range(y, y+2):
                    for i in range(x, x+2):
                        d_min = min(d_min, depth[j][i])
                        d_max = max(d_max, depth[j][i])
                if ((f_px is not None and d_max/d_min > 1.025)
                    or (f_px is None and d_max - d_min > (depth_max - depth_min) * 0.05)):
                    continue
                f.write(f'f {y*width+x+1} {(y+1)*width+x+1} {y*width+(x+1)+1}\n')
                f.write(f'f {y*width+(x+1)+1} {(y+1)*width+x+1} {(y+1)*width+(x+1)+1}\n')

def output_image(depth, is_inverted, filename):
    # Ensure to use image coordinates
    if is_inverted:
        depth = 1.0 / depth
    depth_min = depth.min()
    depth_max = depth.max()

    out = (depth - depth_min) / (depth_max - depth_min)
    cmap = Colormap('inferno')
    out = (cmap(out)[..., :3] * 255.0).astype(np.uint8)

    Image.fromarray(out).save(filename)

def output_stereogram(depth, is_inverted, filename, amplitude):
    width = depth.shape[1]
    height = depth.shape[0]
    out = np.zeros((height, width, 3), np.uint8)

    # Ensure to use image coordinates
    if is_inverted:
        depth = 1.0 / depth
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

    output, image, f_px = model.extract_depth(args.input, args.resize_scale)
    is_inverted = model.is_inverted

    del(model)
    if args.output_format == 'image':
        output_image(output, is_inverted, args.output)
    elif args.output_format == 'mesh':
        output_mesh(output, image, f_px, args.output)
    elif args.output_format == 'stereogram':
        output_stereogram(output, is_inverted, args.output, args.stereo_amplitude)

if __name__ == '__main__':
    main()
