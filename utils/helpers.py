from config import MAX_DESCRIPTION_LENGTH

def truncate_description(text, max_length=MAX_DESCRIPTION_LENGTH):
    """Truncate description to maximum length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text