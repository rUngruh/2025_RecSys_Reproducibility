import ast

def safe_literal_eval(val):
    if val is None:
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Handle any case where literal_eval might fail (if needed)
        return val