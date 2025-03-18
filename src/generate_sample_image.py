from PIL import Image, ImageDraw, ImageFont

def generate_sample_image(text, output_path):
    """
    Generates a synthetic image with the given text.
    Adjust spacing so that each token (e.g., 'gb') is properly handled.
    """
    # Create a blank white image
    width = 800
    height = 100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Try to load a TrueType font, fall back to default if not available.
    try:
        # Provide full path to a font that supports diacritics if necessary.
        font = ImageFont.truetype("arial.ttf", 48)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw text with spacing between tokens
    x = 10
    y = 20
    for token in text.split(" "):  # Tokens separated by spaces
        draw.text((x, y), token, font=font, fill="black")
        # Use textbbox to get the width of the token
        bbox = draw.textbbox((x, y), token, font=font)
        token_width = bbox[2] - bbox[0]
        x += token_width + 20  # Add extra space between tokens
    img.save(output_path)
    print(f"Sample image saved to {output_path}")

if __name__ == '__main__':
    # Sample text: adjust spacing so compound tokens like 'gb' stay together
    sample_text = "à gb à"  # Here, 'gb' is a single token.
    generate_sample_image(sample_text, "sample_test.png")
