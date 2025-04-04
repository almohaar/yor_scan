import cv2
import numpy as np
import tensorflow as tf
from segmentation import segment_document
from config import IMG_HEIGHT, IMG_WIDTH, USE_TESSERACT, TESSERACT_CONFIG
from utils import idx_to_char, combine_lines

# Optional: Use pytesseract as a fallback OCR engine
try:
    import pytesseract
except ImportError:
    pytesseract = None

def preprocess_character_image(char_img):
    """
    Preprocess a segmented character image for model prediction.
    Resizes the image to (IMG_WIDTH, IMG_HEIGHT), converts to RGB, and normalizes pixel values.
    """
    resized = cv2.resize(char_img, (IMG_WIDTH, IMG_HEIGHT))
    # Convert grayscale to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    resized = resized.astype('float32') / 255.0
    return np.expand_dims(resized, axis=0)

def recognize_text(image_path, model_path='models/yoruba_char_model.h5', debug=False):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Read and process the image for segmentation
    document_chars = segment_document(image_path, debug=debug)
    if not document_chars or all(len(line) == 0 for line in document_chars):
        print("No text segments found with segmentation.")
        # Fallback to Tesseract if enabled
        if USE_TESSERACT and pytesseract:
            print("Falling back to Tesseract OCR...")
            img = cv2.imread(image_path)
            text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            return text
        return ""
    
    recognized_lines = []
    # Process each line separately
    for char_images in document_chars:
        line_text = ""
        for char_img in char_images:
            processed_img = preprocess_character_image(char_img)
            prediction = model.predict(processed_img)
            pred_class = prediction.argmax(axis=1)[0]
            line_text += idx_to_char(pred_class)
        recognized_lines.append(line_text)
    
    return combine_lines(recognized_lines)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py <scanned_image_path>")
    else:
        result = recognize_text(sys.argv[1], debug=True)
        print("Recognized Text:\n", result)
