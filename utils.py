import re

def extract_python_code(text):
    """
    Extracts purely the python code from the LLM's markdown output.
    """
    matches = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback: if the model didn't use markdown, return the raw text
    return text.strip()
