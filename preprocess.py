import cv2
import pytesseract
import os
import re

# def preprocess_text(text):
#     # Remove spaces and convert to lowercase
#     preprocessed_text = text.replace(" ", "").lower()
#     preprocess_text_new = re.sub(r'\W+', '', preprocessed_text.replace(" ", "").lower())
#     return preprocess_text_new

def preprocess_text(text):
    # Remove special characters except for spaces, then split into words and join back with single spaces
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    preprocessed_text = ' '.join(words)
    return preprocessed_text


def extract_text(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    preprocessed_text = preprocess_text(text)
    return preprocessed_text

def is_vehicle_ad(text):
    vehicle_keywords = [
        'petrol', 'mileage', 'truck', 'motorcycle', 'bike', 'mercedes', 'benz', 
        'mazda', 'nissan', 'audi', 'hyundai', 'honda', 'toyota', 'rangerover', 'range rover' 
        'ford', 'bmw', 'chevrolet', 'jeep', 'kia', 'lexus', 'land rover', 'landrover',
        'volkswagen', 'volvo', 'suv', 'sedan', 'coupe', 'hatchback', 'bus' 
        'minivan', 'engine', 'horsepower', 'micro', 'full option', 'fulloption', 'hire'
        'hybrid', 'v8', 'v6', '4wheel', '4 wheel', 'leatherseats', 'leather seats', 'buses', 'seater'
        'test drive', 'testdrive', 'vehicle wanted', 'vehiclewanted', 'diesel', 'cruise control', 'cruisecontrol', 'steering wheel', 'steeringwheel', '4wd', 'sun roof', 'sunroof'
        ]
    text = text.lower()
    for keyword in vehicle_keywords:
        if keyword in text:
            return True, keyword
    return False, None

def process_cropped_images(cropped_dir):
    bbox_file_path = os.path.join(cropped_dir, "bboxes.txt")
    vehicle_ad_bboxes = []
    vehicle_ad_files = []
    non_va_bboxes = []
    non_va_files = []
    with open(bbox_file_path, 'r') as bbox_file:
        for line in bbox_file:
            parts = line.strip().split()
            file_name = parts[0]
            x_min, y_min, x_max, y_max = map(int, parts[1:])
            cropped_image_path = os.path.join(cropped_dir, file_name)
            cropped_image = cv2.imread(cropped_image_path)
            text = extract_text(cropped_image)
            is_vehicle, keyword = is_vehicle_ad(text)
            if is_vehicle:
                vehicle_ad_bboxes.append((x_min, y_min, x_max, y_max))
                vehicle_ad_files.append((file_name, x_min, y_min, x_max, y_max, keyword))
            else:
                non_va_bboxes.append((x_min, y_min, x_max, y_max))
                non_va_files.append((file_name, x_min, y_min, x_max, y_max))
    return vehicle_ad_bboxes, vehicle_ad_files, non_va_bboxes, non_va_files