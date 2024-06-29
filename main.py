import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import supervision as sv
from PIL import Image
import pytesseract
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
from transformers import DetrImageProcessor
from preprocess import process_cropped_images
from draw_bbox import draw_bounding_boxes
from save_cropped_va import save_cropped_vehicle_ads
from enhance_image import enhance_image
from save_cropped_images import save_cropped_images
from pdf_to_jpg import pdf_to_jpg


def main(image_path):

    # # Convert PDF to JPG images
    # output_folder = "pdf_images"
    # pdf_to_jpg(pdf_path, output_folder)

    # Define DEVICE
    # DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    DEVICE = torch.device('cpu')

    # loading model

    MODEL_PATH = 'custom-model'
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    # Directory to save the cropped advertisement images (extra)
    OUTPUT_DIR = 'cropped_ads'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)



    # # ------------------ start -----------------------

    id2label = {0: 'ad-cK9v', 1: 'ad-cK9v'}

    box_annotator = sv.BoxAnnotator()

    # Annotate detections
    CONFIDENCE_THRESHOLD = 0.50

    # Ask user for the image path
    # image_path = "./7.jpg"

    # Enhance the entire image
    enhanced_image = enhance_image(image_path)



    # for filename in os.listdir(output_folder):
    #     if filename.endswith(".jpg"):
    #         image_path = os.path.join(output_folder, filename)

    #         # Enhance the entire image
    #         enhanced_image = enhance_image(image_path)

    #         with torch.no_grad():
    #             # Load image and predict
    #             image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    #             inputs = image_processor(images=enhanced_image, return_tensors='pt').to(DEVICE)
    #             outputs = model(**inputs)

    #             # Post-process
    #             target_sizes = torch.tensor([enhanced_image.shape[:2]]).to(DEVICE)
    #             results = image_processor.post_process_object_detection(
    #                 outputs=outputs, 
    #                 threshold=CONFIDENCE_THRESHOLD, 
    #                 target_sizes=target_sizes
    #             )[0]

    #             detections = sv.Detections.from_transformers(transformers_results=results)
                
    #             labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    #             frame_detections = box_annotator.annotate(scene=enhanced_image.copy(), detections=detections, labels=labels)

    #         # Save cropped images of detected advertisements
    #         save_cropped_images(enhanced_image, detections, id2label, OUTPUT_DIR)

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
        
        labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
        frame_detections = box_annotator.annotate(scene=enhanced_image.copy(), detections=detections, labels=labels)

    # Save cropped images of detected advertisements
    save_cropped_images(enhanced_image, detections, id2label, OUTPUT_DIR)

    # ------------- end ----------------------------

    # Directory where cropped images are saved
    cropped_dir = 'cropped_ads'

    output_subfolder = 'vehicle_ads'

    # Process the cropped images and get the bounding boxes and filenames of vehicle ads
    vehicle_ad_bboxes, vehicle_ad_files = process_cropped_images(cropped_dir)

    # Draw bounding boxes on the original image
    image_with_bboxes = draw_bounding_boxes(original_image.copy(), vehicle_ad_bboxes)

    # Save detected vehicle ads
    save_cropped_vehicle_ads(original_image, vehicle_ad_files, cropped_dir, output_subfolder)

    # Construct the file name and path
    file_name = f"detected_{i}.jpg"
    file_path = os.path.join('cropped_ads', file_name)

    cv2.imwrite(file_path, image_with_bboxes)
    print(f"Saved {file_path}")


    i = 0

    # for filename in os.listdir(output_folder):
    #     if filename.endswith(".jpg"):
    #         # image_path = os.path.join(cropped_dir, filename)  # Update with the path to your image
    #         image_path = os.path.join(output_folder, filename)
    #         print(image_path)
    #         original_image = cv2.imread(image_path)
    #         # cv2.imshow("Original Image", original_image)

    #         # Process the cropped images and get the bounding boxes and filenames of vehicle ads
    #         vehicle_ad_bboxes, vehicle_ad_files = process_cropped_images(cropped_dir)

    #         # Draw bounding boxes on the original image
    #         image_with_bboxes = draw_bounding_boxes(original_image.copy(), vehicle_ad_bboxes)

    #         # Save detected vehicle ads
    #         save_cropped_vehicle_ads(original_image, vehicle_ad_files, cropped_dir, output_subfolder)

    #         i += 1

    #         # Construct the file name and path
    #         file_name = f"detected_{i}.jpg"
    #         file_path = os.path.join('cropped_ads', file_name)

    #         cv2.imwrite(file_path, image_with_bboxes)
    #         print(f"Saved {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image to detect vehicle ads.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    args = parser.parse_args()

    main(args.image)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process a pdf to detect vehicle ads.")
#     parser.add_argument("--pdf", required=True, help="Path to the input pdf.")
#     args = parser.parse_args()

# main(args.pdf)