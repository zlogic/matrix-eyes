import torch
from .depth_pro import create_model_and_transforms, load_rgb

class DepthPro:
    def load_model(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        model, transform = create_model_and_transforms(device=device, precision=torch.float32)
        model.eval()
    
        self.device = device
        self.transform = transform
        self.model = model
        self.is_inverted = True
    
    def extract_depth(self, image_path, resize_scale):
        image, _, f_px = load_rgb(image_path, resize_scale)

        prediction = self.model.infer(self.transform(image), f_px=f_px)

        # Depth in meters;
        # as image coordinates are projected values (in pixels), real coordinates are x*z/f_px
        depth = prediction['depth'].detach().cpu().numpy().squeeze()
        return depth, image, f_px
    
    def __init__(self):
        self.load_model()
