import cv2
# import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from enhance_image import multi_scale_sharpen

def save_cropped_images(image, detections, id2label, output_dir, padding=20):
    """Save cropped images of detected advertisements and their bounding boxes."""
    bbox_file_path = os.path.join(output_dir, "bboxes.txt")
    with open(bbox_file_path, 'w') as bbox_file:
        for i, (bbox, confidence, class_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            if bbox is not None:
                label = id2label[class_id.item()]
                # if label.lower() == 'ad-cK9v':
                if label == 'ad-cK9v':
                    x_min, y_min, x_max, y_max = bbox.astype(int)

                    # Adding padding to the bounding box
                    x_min = max(x_min - padding, 0)
                    y_min = max(y_min - padding, 0)
                    x_max = min(x_max + padding, image.shape[1])
                    y_max = min(y_max + padding, image.shape[0])

                    # Cropping the image
                    cropped_image = image[y_min:y_max, x_min:x_max]

                    # Sharpen the cropped image
                    cropped_image = multi_scale_sharpen(cropped_image)

                    # Optional: Rescale cropped image if needed
                    upscale_factor = 1
                    cropped_image = cv2.resize(cropped_image, 
                                               (cropped_image.shape[1] * upscale_factor, cropped_image.shape[0] * upscale_factor),
                                               interpolation=cv2.INTER_CUBIC)

                    # Construct the file name and path
                    file_name = f"{label}_{i}_conf_{confidence:.2f}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # Save the cropped image
                    cv2.imwrite(file_path, cropped_image)
                    # print(f"Saved {file_path}")

                    # Save the bounding box information
                    bbox_file.write(f"{file_name} {x_min} {y_min} {x_max} {y_max}\n")
            else:
                print(f"Skipping detection {i} as bbox is None")