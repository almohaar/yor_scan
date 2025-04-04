# Image dimensions for training the character classifier
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 20

# Define the Yoruba character list as tokens.
# Ensure your dataset folder names match these tokens exactly.
CHAR_LIST = [
    "a", "à", "á", "b", "d", "e", "è", "é", "ẹ", "f", "gb", "h", "i", "ì", "í", "j", "k", "l", "m", "n", "o", "ò", "ó", "ọ", "ṣ", "p", "r", "s", "t", "u", "ù", "ú", "w", "y",
    "A", "À", "Á", "B", "D", "E", "È", "É", "Ẹ", "F", "GB", "H", "I", "Ì", "Í", "J", "K", "L", "M", "N", "O", "Ò", "Ó", "Ọ", "R", "S", "Ṣ", "T", "U", "Ù", "Ú", "W", "Y"
]

ROTATION_FACTOR = 0.1  # Radians
ZOOM_FACTOR = 0.1

# Tesseract configuration (if fallback is used)
USE_TESSERACT = True
TESSERACT_CONFIG = '--psm 6'  # Assume a single uniform block of text