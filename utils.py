import cv2
import numpy as np

def load_and_preprocess(image_path, target_size=(128, 128)):
    """Load and preprocess an image."""
    img = cv2.imread(image_path, 0)  # Grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, target_size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img