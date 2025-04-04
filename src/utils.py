# Utility functions for OCR Yoruba project

# Example mapping of class indices to Yoruba characters.
# Update the dictionary with your full set of Yoruba characters.
CHAR_DICT = {
    0: 'a', 1: 'b', 2: 'd', 3: 'e', 4: 'f', 5: 'g', 6: 'h', 7: 'i',
    8: 'j', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'r',
    16: 's', 17: 't', 18: 'u', 19: 'w', 20: 'y',
    # Include uppercase and special Yoruba diacritics if needed
}

def idx_to_char(idx):
    """Converts a model prediction index to a character."""
    return CHAR_DICT.get(idx, '')

def combine_lines(line_chars):
    """
    Given a list of lists of recognized characters (each inner list is one line),
    returns the full text with line breaks.
    """
    return "\n".join(["".join(chars) for chars in line_chars])
