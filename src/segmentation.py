import cv2

def segment_characters(image_path, min_area=10, debug=False):
    """
    Segments a scanned document image into individual character images.
    
    Parameters:
      image_path: Path to the scanned image.
      min_area: Minimum area (in pixels) for a contour to be considered a character.
      debug: If True, saves the thresholded image for debugging.
      
    Returns:
      A list of character images (cropped from the original image), sorted from left to right.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return []
    
    # Apply thresholding to obtain a binary image
    # (Adjust threshold value or method if needed)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    if debug:
        cv2.imwrite("debug_threshold.png", thresh)
        print("Saved thresholded image as debug_threshold.png")
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue  # Filter out small noise
        roi = thresh[y:y+h, x:x+w]
        char_regions.append((x, roi))
    
    # Sort regions by the x-coordinate (left-to-right order)
    char_regions = sorted(char_regions, key=lambda b: b[0])
    
    return [roi for (x, roi) in char_regions]
