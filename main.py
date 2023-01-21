import argparse
import torch
import cv2
import numpy as np

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

def output_image(depth, filename):
    depth_min = depth.min()
    depth_max = depth.max()

    out = 255 * (depth - depth_min) / (depth_max - depth_min)
    out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
    
    cv2.imwrite(filename, out.astype("uint8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type',
        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
        default='DPT_Large')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    midas = MiDaS(args.model_type)
    output = midas.extract_depth(img)

    del(midas)
    output_image(output, args.output)

if __name__ == '__main__':
    main()
