import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def extract_python_code(text):
    """
    Extracts purely the python code from the LLM's markdown output.
    """
    matches = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    
    # Fallback: if the model didn't use markdown, return the raw text
    return text.strip()


def is_degenerate_output(text, repeated_line_threshold=4, ngram_size=8, repeated_ngram_threshold=3):
    normalized_text = text.strip()
    if not normalized_text:
        return False

    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    longest_line_run = 1
    current_line_run = 1

    for previous_line, current_line in zip(lines, lines[1:]):
        if current_line == previous_line:
            current_line_run += 1
            longest_line_run = max(longest_line_run, current_line_run)
        else:
            current_line_run = 1

    if longest_line_run >= repeated_line_threshold:
        return True

    tokens = TOKEN_PATTERN.findall(normalized_text.lower())
    min_tokens = ngram_size * repeated_ngram_threshold
    if len(tokens) < min_tokens:
        return False

    upper_bound = len(tokens) - min_tokens + 1
    for start_index in range(upper_bound):
        window = tokens[start_index:start_index + ngram_size]
        repeat_count = 1
        cursor = start_index + ngram_size

        while cursor + ngram_size <= len(tokens):
            if tokens[cursor:cursor + ngram_size] != window:
                break

            repeat_count += 1
            if repeat_count >= repeated_ngram_threshold:
                return True
            cursor += ngram_size

    return False
