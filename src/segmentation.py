import cv2
import numpy as np

def segment_lines(image, debug=False):
    """
    Segment the full document image into lines using horizontal projection.
    
    Parameters:
      image: Grayscale input image.
      debug: If True, saves intermediate images for debugging.
      
    Returns:
      List of line image crops.
    """
    # Apply a binary threshold. Using adaptive threshold for varying lighting.
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    if debug:
        cv2.imwrite("debug_adaptive_threshold.png", thresh)
    
    # Use dilation to connect letters into lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    if debug:
        cv2.imwrite("debug_dilated_lines.png", dilated)
    
    # Find contours corresponding to text lines.
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    # Sort contours top-to-bottom
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])
    for (x, y, w, h) in bounding_boxes:
        # Crop the line region from the original thresholded image.
        line_img = thresh[y:y+h, x:x+w]
        lines.append(line_img)
    return lines

def segment_characters(line_img, min_area=50, debug=False, line_index=0):
    """
    Segments a line image into individual character images.
    
    Parameters:
      line_img: Thresholded image of a single text line.
      min_area: Minimum area for a contour to be considered a character.
      debug: If True, saves intermediate images for debugging.
      line_index: Index of the current line (for debug file naming).
      
    Returns:
      A list of character images (sorted left-to-right).
    """
    # Dilate a bit to ensure characters are connected well.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(line_img, kernel, iterations=1)
    
    if debug:
        cv2.imwrite(f"debug_line_{line_index}_dilated.png", processed)
    
    # Find contours of potential characters.
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue  # Ignore small noise regions
        roi = line_img[y:y+h, x:x+w]
        char_regions.append((x, roi))
    
    # Sort the detected characters left-to-right.
    char_regions = sorted(char_regions, key=lambda b: b[0])
    return [roi for (x, roi) in char_regions]

def segment_document(image_path, debug=False):
    """
    Segments the entire document into lines and then into characters.
    
    Parameters:
      image_path: Path to the scanned image.
      debug: If True, saves intermediate images.
      
    Returns:
      List of lists. Each inner list contains character images for that line.
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return []
    
    # Segment into lines
    lines = segment_lines(img, debug=debug)
    if not lines:
        print("No text lines found.")
        return []
    
    document_chars = []
    for idx, line in enumerate(lines):
        chars = segment_characters(line, debug=debug, line_index=idx)
        document_chars.append(chars)
    return document_chars
