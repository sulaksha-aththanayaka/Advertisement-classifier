import cv2

def draw_bounding_boxes(image, bboxes):
    for (x_min, y_min, x_max, y_max) in bboxes:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image