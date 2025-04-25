import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(left_img, disparity_map, depth_map, output_path="D:/code/kitti_raft_stereo_depth_results.png"):
    """Hiển thị và lưu ảnh: ảnh trái, bản đồ disparity, bản đồ độ sâu."""
    # Chuẩn hóa disparity để hiển thị
    norm_disparity_map = 255 * ((disparity_map - np.min(disparity_map)) /
                                (np.max(disparity_map) - np.min(disparity_map) + 1e-6))
    color_disparity = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, alpha=1), cv2.COLORMAP_JET)
    
    # Chuẩn hóa depth để hiển thị, giới hạn tối đa 100m
    max_dist = 100
    depth_map[depth_map > max_dist] = max_dist  # Giới hạn độ sâu
    norm_depth_map = 255 * (1 - depth_map / max_dist)
    norm_depth_map[norm_depth_map < 0] = 0
    norm_depth_map[norm_depth_map >= 255] = 0
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, alpha=1), cv2.COLORMAP_JET)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Left Image")
    plt.imshow(left_img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Disparity Map")
    plt.imshow(color_disparity)
    plt.colorbar(label='Disparity')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Depth Map")
    plt.imshow(color_depth)
    plt.colorbar(label='Depth (m)')
    plt.axis('off')
    
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Save as: {output_path}")
    except Exception as e:
        print(f"Eror: {str(e)}")
    
    # Hiển thị trên màn hình
    plt.show()
    plt.pause(5)  # Giữ cửa sổ mở 5 giây, sau đó tự đóng
    plt.close()