#Image enhancing functions

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# Function to apply sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Function to denoise the image
def denoise_image(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Function to enhance contrast using CLAHE
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

# Function to enhance image
def enhance_image(image_path):
    image = cv2.imread(image_path)
    
    # Apply sharpening
    sharpened_image = sharpen_image(image)
    
    # Apply denoising
    denoised_image = denoise_image(sharpened_image)
    
    # Enhance contrast
    final_image = enhance_contrast(denoised_image)
    
    return final_image


# Function to display images
def display_images(original_image, enhanced_image):
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Enhanced Image')
    plt.imshow(enhanced_image_rgb)
    plt.axis('off')
    plt.show()

# extra
def multi_scale_sharpen(image, scales=[(3, 1.0), (5, 1.5)], amount=0.1):
    """Apply multi-scale sharpening to the image."""
    sharpened = image.copy().astype(np.float32)
    
    for kernel_size, sigma in scales:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        mask = image - blurred
        sharpened += amount * mask
    
    # Clip the values to [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened