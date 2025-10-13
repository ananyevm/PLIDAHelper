from config import MAX_DESCRIPTION_LENGTH

def truncate_description(text, max_length=MAX_DESCRIPTION_LENGTH, query_analyzer=None, context_type="dataset"):
    """Generate a concise description using OpenAI, or truncate if unavailable."""
    # If query_analyzer is provided and text is long, use OpenAI to generate concise description
    if query_analyzer and len(text) > max_length:
        try:
            return query_analyzer.generate_concise_description(text, context_type)
        except Exception:
            # Fallback to truncation if OpenAI fails
            pass
    
    # Default truncation behavior
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text