import re

def extract_python_code(text):
    """
    Extracts purely the python code from the LLM's markdown output.
    """
    match = re.search(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: if the model didn't use markdown, return the raw text
    return text.strip()
