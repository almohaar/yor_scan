import cv2
import numpy as np
import tensorflow as tf
from segmentation import segment_characters
from config import IMG_HEIGHT, IMG_WIDTH
from utils import idx_to_char

def preprocess_character_image(char_img):
    """
    Preprocess the segmented character image for model prediction.
    Resizes the image to (IMG_WIDTH, IMG_HEIGHT), converts to RGB, and normalizes pixel values.
    """
    resized = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))
    # Convert grayscale to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    resized = resized.astype('float32') / 255.0
    return np.expand_dims(resized, axis=0)

def recognize_text(image_path, model_path='models/yoruba_char_model.h5'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Segment the image into character images (set debug=True to save threshold image)
    char_images = segment_characters(image_path, min_area=10, debug=True)
    if not char_images:
        print("No character regions found.")
        return ""
    
    recognized_text = ""
    for char_img in char_images:
        processed_img = preprocess_character_image(char_img)
        prediction = model.predict(processed_img)
        pred_class = prediction.argmax(axis=1)[0]
        recognized_text += idx_to_char(pred_class)
    
    return recognized_text

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <scanned_image_path>")
    else:
        result = recognize_text(sys.argv[1])
        print("Recognized Text:", result)
