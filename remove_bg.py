import numpy as np
import cv2

def remove_non_vehicle_ads(original_image_path, bbox_file_path):
    # Load original image
    original_image = cv2.imread(original_image_path)
    
    # Read bounding boxes from bbox_file_path
    vehicle_ad_bboxes = []
    with open(bbox_file_path, 'r') as bbox_file:
        for line in bbox_file:
            parts = line.strip().split()
            x_min, y_min, x_max, y_max = map(int, parts[1:])
            vehicle_ad_bboxes.append((x_min, y_min, x_max, y_max))
    
    # Create mask to keep only regions inside bounding boxes
    mask = np.zeros_like(original_image)
    for (x_min, y_min, x_max, y_max) in vehicle_ad_bboxes:
        mask[y_min:y_max, x_min:x_max] = original_image[y_min:y_max, x_min:x_max]
    
    # Apply mask to original image
    result_image = cv2.bitwise_and(original_image, mask)
    
    return result_image