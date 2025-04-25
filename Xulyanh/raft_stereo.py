import torch
import numpy as np
import argparse
import sys
import os

# Add cloned repo path (update to your repo location if different)
sys.path.append(os.path.abspath("depth_estimation_stereo_images"))
from networks.RAFTStereo.core.raft_stereo import RAFTStereo
from networks.RAFTStereo.core.utils.utils import InputPadder

class RAFTStereoModel:
    def __init__(self, model_path):
        """Initialize RAFT-Stereo model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = self.get_internal_args()
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.module.to(self.device)
        self.model.eval()
        print(f"RAFT-Stereo model loaded from: {model_path}, Device: {self.device}")

    def get_internal_args(self):
        """Parse RAFT-Stereo arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mixed_precision', action='store_true')
        parser.add_argument('--valid_iters', type=int, default=32)
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
        parser.add_argument('--shared_backbone', action='store_true')
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--n_downsample', type=int, default=2)
        parser.add_argument('--slow_fast_gru', action='store_true')
        parser.add_argument('--n_gru_layers', type=int, default=3)
        return parser.parse_args()

    def load_image(self, img):
        """Convert image to tensor."""
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def compute_disparity(self, left_img, right_img):
        """Compute disparity map using RAFT-Stereo."""
        with torch.no_grad():
            image1 = self.load_image(left_img)
            image2 = self.load_image(right_img)
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            
            _, flow_up = self.model(image1, image2, iters=self.get_internal_args().valid_iters, test_mode=True)
            disparity_map = -flow_up.cpu().numpy().squeeze()
        
        return disparity_map