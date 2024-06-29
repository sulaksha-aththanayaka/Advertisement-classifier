import cv2
import numpy as np
import os
import torch
import supervision as sv
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from preprocess import process_cropped_images
from draw_bbox import draw_bounding_boxes
from save_cropped_va import save_cropped_vehicle_ads, save_cropped_non_vehicle_ads
from enhance_image import enhance_image
from save_cropped_images import save_cropped_images
import argparse

# Define DEVICE
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
MODEL_PATH = 'custom-model'
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# Directory to save the cropped advertisement images
OUTPUT_DIR = 'cropped_ads'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping of IDs to labels
id2label = {0: 'ad-cK9v', 1: 'ad-cK9v'}

box_annotator = sv.BoxAnnotator()

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.60

def main(image_path):
    # Enhance the entire image
    enhanced_image = enhance_image(image_path)

    # Annotate detections
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
        labels = [f"{id2label[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        frame_detections = box_annotator.annotate(scene=enhanced_image.copy(), detections=detections, labels=labels)

    # Save cropped images of detected advertisements
    save_cropped_images(enhanced_image, detections, id2label, OUTPUT_DIR)

    # Directory where cropped images are saved
    cropped_dir = 'cropped_ads'
    output_subfolder_1 = 'vehicle_ads'
    output_subfolder_2 = 'non_vehicle_ads'
    vehicle_ads_dir = os.path.join(cropped_dir, 'vehicle_ads')
    bbox_file_path = os.path.join(cropped_dir, 'bboxes.txt')

    # Load original image
    original_image = cv2.imread(image_path)

    # Process the cropped images and get the bounding boxes and filenames of vehicle ads
    vehicle_ad_bboxes, vehicle_ad_files, non_va_bboxes, non_va_files = process_cropped_images(cropped_dir)

    # Draw bounding boxes on the original image
    image_with_bboxes = draw_bounding_boxes(original_image.copy(), vehicle_ad_bboxes)

    # Save detected vehicle ads
    save_cropped_vehicle_ads(original_image, vehicle_ad_files, cropped_dir, output_subfolder_1)

    # Save detected non-vehicle ads
    save_cropped_non_vehicle_ads(original_image, non_va_files, cropped_dir, output_subfolder_2)

    # Read the filenames of vehicle ads
    vehicle_ad_filenames = read_vehicle_ad_filenames(vehicle_ads_dir)

    # Parse the bbox.txt file to get bounding box information
    bbox_info = parse_bbox_file(bbox_file_path)

    # Filter bounding box information for vehicle ads
    vehicle_bboxes = filter_vehicle_bboxes(vehicle_ad_filenames, bbox_info)

    mask = np.ones_like(original_image, dtype=np.uint8) * 255

    for filename, bbox in vehicle_bboxes.items():
        x_min, y_min, x_max, y_max = bbox
        mask[y_min:y_max, x_min:x_max] = original_image[y_min:y_max, x_min:x_max]

    # Construct the file name and path
    file_name = "detected.jpg"
    file_path = os.path.join('detected_ads', file_name)

    # Save the image with removed non-vehicle ad regions
    cv2.imwrite(file_path, mask)
    print(f"Saved {file_path}")

def read_vehicle_ad_filenames(vehicle_ads_dir):
    """Read the filenames of vehicle ads from the specified directory."""
    vehicle_ad_filenames = [filename for filename in os.listdir(vehicle_ads_dir) if filename.endswith('.jpg')]
    return vehicle_ad_filenames

def parse_bbox_file(bbox_file_path):
    """Parse the bbox.txt file to get bounding box information."""
    bbox_info = {}
    with open(bbox_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            filename = parts[0]
            bbox = list(map(int, parts[1:]))
            bbox_info[filename] = bbox
    return bbox_info

def filter_vehicle_bboxes(vehicle_ad_filenames, bbox_info):
    """Filter bounding box information for vehicle ads."""
    vehicle_bboxes = {}
    for va_filename in vehicle_ad_filenames:
        base_filename = va_filename.split('_', 1)[1]  # Split by the first '_' and take the second part
        if base_filename in bbox_info:
            vehicle_bboxes[va_filename] = bbox_info[base_filename]
    return vehicle_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to detect advertisements.')
    parser.add_argument('--image', required=True, help='Path to the input image')
    args = parser.parse_args()
    main(args.image)
