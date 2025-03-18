from config import CHAR_LIST

def idx_to_char(index):
    """Convert a numeric index to its corresponding character token."""
    if 0 <= index < len(CHAR_LIST):
        return CHAR_LIST[index]
    return ''

def char_to_idx(char):
    """Convert a character token to its numeric index.
    This ensures tokens like 'gb' are handled as a single unit.
    """
    try:
        return CHAR_LIST.index(char)
    except ValueError:
        return -1
