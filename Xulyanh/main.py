from pathlib import Path
from data_loader import load_kitti_images, get_calibration_parameters
from raft_stereo import RAFTStereoModel
from depth_estimator import calc_depth_map
from visualizer import visualize_results

def main(left_path, right_path, calib_path=None, model_path="pretrained_models/raft_stereo/raft-stereo_20000.pth"):
    """Main pipeline for stereo depth estimation."""
    # Load images
    left_img, right_img = load_kitti_images(left_path, right_path)
    
    # Load calibration
    parameters = get_calibration_parameters(calib_path)
    k_left, t_left, p_left, k_right, t_right, p_right = parameters
    print(f"Focal Length: {k_left[0, 0]} pixels, Baseline: {abs(t_left[0] - t_right[0])} meters")
    
    # Initialize RAFT-Stereo model
    stereo_model = RAFTStereoModel(model_path)
    
    # Compute disparity map
    disparity = stereo_model.compute_disparity(left_img, right_img)
    
    # Compute depth map
    depth = calc_depth_map(disparity, k_left, t_left, t_right)
    
    # Visualize results
    visualize_results(left_img, disparity, depth)
    
    return disparity, depth

if __name__ == "__main__":
    # Your image paths
    left_image_path = Path("left_image.jpg")
    right_image_path = Path("right_image.jpg")
    
    # Model path
    model_path = Path("pretrained_models/raft_stereo/raft-stereo_20000.pth")
    
    # Calibration path (None uses defaults)
    calib_path = None  # Update if you have a calib file, e.g., "D:/code/calib_cam_to_cam.txt"
    
    # Check files
    if not (left_image_path.exists() and right_image_path.exists()):
        print("Image files not found. Please check paths.")
    elif not model_path.exists():
        print(f"Model weights not found at {model_path}. Please ensure RAFT-Stereo weights are available.")
    else:
        disparity_map, depth_map = main(left_image_path, right_image_path, calib_path, model_path)