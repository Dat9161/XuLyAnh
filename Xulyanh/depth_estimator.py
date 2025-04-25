import numpy as np

def calc_depth_map(disp_left, k_left, t_left, t_right):
    """Compute depth map from disparity."""
    f = k_left[0, 0]  # Focal length
    b = abs(t_left[0] - t_right[0])  # Baseline
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    depth_map = np.ones(disp_left.shape, np.single)
    depth_map[:] = f * b / disp_left[:]
    return depth_map
