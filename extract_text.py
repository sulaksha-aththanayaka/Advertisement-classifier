import cv2
import pytesseract
import os
import re

def extract_text(image):

    # Read the image from the file
    image = cv2.imread(image)
    
    # Check if the image is successfully loaded
    if image is None:
        raise FileNotFoundError(f"Image at path '{image}' not found or unable to read.")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    preprocessed_text = preprocess_text(text)
    return preprocessed_text

def preprocess_text(text):
    # Remove special characters except for spaces, then split into words and join back with single spaces
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# def preprocess_text(text):
#     # Remove spaces and convert to lowercase
#     # preprocessed_text = text.replace(" ", "").lower()
#     preprocess_text_new = re.sub(r'\W+', '', text.replace(" ", "").lower())
#     return preprocess_text_new


def is_vehicle_ad(text):
    vehicle_keywords = [
        'petrol', 'mileage', 'truck', 'motorcycle', 'bike', 'mercedes', 'benz', 
        'mazda', 'nissan', 'audi', 'hyundai', 'honda', 'toyota', 'mitsubishi', 'rangerover', 'range rover' 
        'ford', 'bmw', 'chevrolet', 'jeep', 'kia', 'lexus', 'land rover', 'landrover',
        'volkswagen', 'volvo', 'suv', 'sedan', 'coupe', 'hatchback', 'car', 'bus' 
        'minivan', 'engine', 'horsepower', 'micro', 'full option', 'fulloption'
        'hybrid', 'v8', 'v6', '4wheel', '4 wheel', 'leatherseats', 'leather seats', 'buses'
        'test drive', 'testdrive', 'vehicle wanted', 'vehiclewanted', 'diesel', 'cruise control', 'cruisecontrol', 'steering wheel', 'steeringwheel', '4wd', 'sun roof', 'sunroof'
        ]
    text = text.lower()
    for keyword in vehicle_keywords:
        if keyword in text:
            return True, keyword
    return False, None

text = extract_text('./test11.jpg')
print(text)