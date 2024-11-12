import torch
import numpy as np
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

    def extract_depth(self, image_path, resize_scale):
        img = Image.open(image_path)

        if resize_scale != 1.0:
            scale = resize_scale
            img = img.resize(int(img.width*scale), int(img.height*scale), resample=Image.Resampling.BICUBIC)

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
