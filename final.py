import cv2
import os
import argparse
import torch
import supervision as sv
from transformers import DetrForObjectDetection, DetrImageProcessor
from preprocess import process_cropped_images
from save_cropped_images import save_cropped_images
import numpy as np
from enhance_image import enhance_image


def main(image_path):
    # Define DEVICE
    DEVICE = torch.device('cpu')  # Change to 'cuda' if GPU is available

    # loading model
    MODEL_PATH = 'custom-model'
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    # Directory to save the cropped advertisement images (extra)
    OUTPUT_DIR = 'cropped_ads'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Ask user for the image path
    # image_path = "./3.jpg"

    # Enhance the entire image
    enhanced_image = enhance_image(image_path)

    # Annotate detections
    CONFIDENCE_THRESHOLD = 0.50

    id2label = {0: 'ad-cK9v', 1: 'ad-cK9v'}

    with torch.no_grad():
        # Load image and predict
        image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        inputs = image_processor(images=enhanced_image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([enhanced_image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=CONFIDENCE_THRESHOLD,
            target_sizes=target_sizes
        )[0]

        detections = sv.Detections.from_transformers(transformers_results=results)

        # Debug print to inspect the structure of `detections`
        # print("Detections structure:", detections)

        # Save cropped images of detected advertisements
        save_cropped_images(enhanced_image, detections, id2label, output_dir=OUTPUT_DIR)


    # Directory where cropped images are saved
    cropped_dir = 'cropped_ads'
    output_subfolder = 'vehicle_ads'
    detected_dir = 'detected_ads'

    # Process the cropped images and get the bounding boxes and filenames of vehicle ads
    vehicle_ad_bboxes, _ = process_cropped_images(cropped_dir)

    # Load original image
    original_image = cv2.imread(image_path)

    # Create mask to retain only vehicle ad regions
    # mask = np.zeros_like(original_image)
    mask = np.ones_like(original_image, dtype=np.uint8) * 255 

    # Apply mask for each vehicle ad bounding box
    for (x_min, y_min, x_max, y_max) in vehicle_ad_bboxes:
        mask[y_min:y_max, x_min:x_max] = original_image[y_min:y_max, x_min:x_max]

    # Construct the file name and path
    file_name = f"detected.jpg"
    file_path = os.path.join(detected_dir ,file_name)

    # Save the image with removed non-vehicle ad regions
    cv2.imwrite(file_path, mask)
    print(f"Saved {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image to detect vehicle ads.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    args = parser.parse_args()

    main(args.image)