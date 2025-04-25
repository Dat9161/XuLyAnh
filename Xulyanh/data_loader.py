import cv2
import numpy as np
from pathlib import Path

def load_kitti_images(left_path, right_path):
    """Load stereo images and convert to RGB."""
    left_path, right_path = Path(left_path), Path(right_path)
    if not (left_path.exists() and right_path.exists()):
        raise FileNotFoundError(f"Images not found: {left_path}, {right_path}")
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    if left_img is None or right_img is None:
        raise ValueError("Failed to load images.")
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    return left_img, right_img

def get_calibration_parameters(calib_path=None):
    """Load KITTI calibration or use defaults."""
    if calib_path and Path(calib_path).exists():
        parameters = []
        with open(calib_path, 'r') as f:
            fin = f.readlines()
            for line in fin:
                if line[:4] == 'K_02':
                    parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
                elif line[:4] == 'T_02':
                    parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
                elif line[:9] == 'P_rect_02':
                    parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1))
                elif line[:4] == 'K_03':
                    parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
                elif line[:4] == 'T_03':
                    parameters.append(np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1))
                elif line[:9] == 'P_rect_03':
                    parameters.append(np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1))
        return parameters
    else:
        # Default KITTI parameters (approx. fx=718, baseline=0.54m)
        k_left = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]], dtype=np.float32)
        t_left = np.array([0, 0, 0], dtype=np.float32)
        p_left = np.array([[718.856, 0, 607.1928, 0], [0, 718.856, 185.2157, 0], [0, 0, 1, 0]], dtype=np.float32)
        k_right = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1]], dtype=np.float32)
        t_right = np.array([-0.54, 0, 0], dtype=np.float32)
        p_right = np.array([[718.856, 0, 607.1928, -386.1448], [0, 718.856, 185.2157, 0], [0, 0, 1, 0]], dtype=np.float32)
        return [k_left, t_left, p_left, k_right, t_right, p_right]
