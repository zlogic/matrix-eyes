import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class MiDaS:
    def load_model(self):
        midas = torch.hub.load('intel-isl/MiDaS', self.model_type)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        midas.to(device)
        midas.eval()

        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

        if self.model_type == 'DPT_Large' or self.model_type == 'DPT_Hybrid':
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.midas = midas
        self.device = device

    def extract_depth(self, img):
        img = np.array(img)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        return output

    def __init__(self, model_type):
        self.model_type = model_type
        self.load_model()

def output_mesh(depth, filename):
    width = depth.shape[1]
    height = depth.shape[0]
    with open(filename, 'w') as f:
        for y in range(height):
            for x in range(width):
                f.write(f'v {x} {height-y} {depth[y][x]}\n')

        for y in range(height-1):
            for x in range(width-1):
                f.write(f'f {y*width+x+1} {(y+1)*width+x+1} {y*width+(x+1)+1}\n')
                f.write(f'f {y*width+(x+1)+1} {(y+1)*width+x+1} {(y+1)*width+(x+1)+1}\n')

def output_image(depth, filename):
    depth_min = depth.min()
    depth_max = depth.max()

    out = (depth - depth_min) / (depth_max - depth_min)
    cmap = plt.get_cmap("inferno")
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
        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
        default='DPT_Large')
    parser.add_argument('--output-format', choices=['image', 'mesh', 'stereogram'], default='image')
    parser.add_argument('--resize-scale', type=float, default=1.0)
    parser.add_argument('--stereo-amplitude', type=float, default=1.0/16)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    img = Image.open(args.input)

    if args.resize_scale != 1.0:
        scale = args.resize_scale
        img = img.resize(int(img.width*scale), int(img.height*scale), resample=Image.Resampling.BICUBIC)

    midas = MiDaS(args.model_type)
    output = midas.extract_depth(img)

    del(midas)
    if args.output_format == 'image':
        output_image(output, args.output)
    elif args.output_format == 'mesh':
        output_mesh(output, args.output)
    elif args.output_format == 'stereogram':
        output_stereogram(output, args.output, args.stereo_amplitude)

if __name__ == '__main__':
    main()
