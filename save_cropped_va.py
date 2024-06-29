import cv2
# import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from enhance_image import multi_scale_sharpen

def save_cropped_vehicle_ads(original_image, vehicle_ad_files, cropped_dir, output_subfolder, padding=10):
    output_dir = os.path.join(cropped_dir, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    keyword_file_path = os.path.join(output_dir, "keywords.txt")
    with open(keyword_file_path, 'w') as keyword_file:
        for i, (file_name, x_min, y_min, x_max, y_max, keyword) in enumerate(vehicle_ad_files):
            # Adding padding to the bounding box
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, original_image.shape[1])
            y_max = min(y_max + padding, original_image.shape[0])
            
            # Cropping the image
            cropped_image = original_image[y_min:y_max, x_min:x_max]

            # Optional: Sharpen the cropped image
            cropped_image = multi_scale_sharpen(cropped_image)

            # Optional: Rescale cropped image if needed
            upscale_factor = 1
            cropped_image = cv2.resize(cropped_image, 
                                       (cropped_image.shape[1] * upscale_factor, cropped_image.shape[0] * upscale_factor),
                                       interpolation=cv2.INTER_CUBIC)

            # Construct the file name and path
            new_file_name = f"{i}_{file_name}"
            file_path = os.path.join(output_dir, new_file_name)
            
            # Save the cropped image
            cv2.imwrite(file_path, cropped_image)
            # print(f"Saved {file_path}")

            # Write keyword information to the text file
            keyword_file.write(f"{new_file_name} - {keyword}\n")

def save_cropped_non_vehicle_ads(original_image, vehicle_ad_files, cropped_dir, output_subfolder, padding=10):
    output_dir = os.path.join(cropped_dir, output_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    keyword_file_path = os.path.join(output_dir, "keywords.txt")
    with open(keyword_file_path, 'w') as keyword_file:
        for i, (file_name, x_min, y_min, x_max, y_max) in enumerate(vehicle_ad_files):
            # Adding padding to the bounding box
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, original_image.shape[1])
            y_max = min(y_max + padding, original_image.shape[0])
            
            # Cropping the image
            cropped_image = original_image[y_min:y_max, x_min:x_max]

            # Optional: Sharpen the cropped image
            cropped_image = multi_scale_sharpen(cropped_image)

            # Optional: Rescale cropped image if needed
            upscale_factor = 1
            cropped_image = cv2.resize(cropped_image, 
                                       (cropped_image.shape[1] * upscale_factor, cropped_image.shape[0] * upscale_factor),
                                       interpolation=cv2.INTER_CUBIC)

            # Construct the file name and path
            new_file_name = f"{i}_{file_name}"
            file_path = os.path.join(output_dir, new_file_name)
            
            # Save the cropped image
            cv2.imwrite(file_path, cropped_image)
            # print(f"Saved {file_path}")

            # Write keyword information to the text file
            keyword_file.write(f"{new_file_name}\n")