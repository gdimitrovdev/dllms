import ast
import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FENCED_CODE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
JUPYTER_CODE_PATTERN = re.compile(r"<jupyter_code>\s*(.*?)(?=<jupyter_output>|<jupyter_text>|<jupyter_code>|$)", re.DOTALL | re.IGNORECASE)
FENCE_TOKEN_PATTERN = re.compile(r"```(?:python)?", re.IGNORECASE)
NOTEBOOK_TAG_PATTERN = re.compile(
    r"</?(?:jupyter_(?:code|output|text)|j(?:up(?:yter)?|yupyter|upupyter)[_\\/\-]?(?:code|output|text|p)?|s(?:[_\\/\-][^>\n]+)?|up)>|<empty_output>|</>",
    re.IGNORECASE,
)
STRUCTURED_LINE_PATTERN = re.compile(r"^\s*(?:from |import |class |def |async def |@)")


def _decode_generated_text(text):
    normalized = text.replace("\u0120", " ").replace("\u010a", "\n")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip()


def _normalize_generated_text(text):
    normalized = _decode_generated_text(text)
    normalized = FENCE_TOKEN_PATTERN.sub("\n", normalized)
    normalized = NOTEBOOK_TAG_PATTERN.sub("\n", normalized)
    return normalized.strip()


def _clean_code_candidate(candidate):
    cleaned = _normalize_generated_text(candidate)
    cleaned_lines = [line.rstrip() for line in cleaned.splitlines()]
    return "\n".join(cleaned_lines).strip()


def _expand_compact_python(candidate):
    if "\n" in candidate or "    " not in candidate:
        return candidate

    expanded = re.sub(r"(?<=\S)( {4,})(?=\S)", lambda match: "\n" + match.group(1), candidate)
    return expanded.strip()


def _looks_like_python(candidate):
    python_markers = ("def ", "class ", "import ", "from ", "return ", "for ", "while ", "if ", "try:")
    return any(marker in candidate for marker in python_markers)


def _python_candidate_score(candidate):
    structured_prefixes = (
        "def ", "class ", "import ", "from ", "return ", "for ", "while ",
        "if ", "elif ", "else:", "try:", "except", "with ", "@",
    )
    code_like_lines = 0
    prose_like_lines = 0

    for line in candidate.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(structured_prefixes) or stripped.endswith(":") or stripped.startswith(("(", "[", "{")):
            code_like_lines += 1
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_\s,().+\-/*=]*$", stripped) and " " in stripped and not _looks_like_python(stripped):
            prose_like_lines += 1
            continue
        code_like_lines += 1

    markup_penalty = candidate.count("```") + len(NOTEBOOK_TAG_PATTERN.findall(candidate))
    return (code_like_lines - prose_like_lines, -markup_penalty, code_like_lines, -prose_like_lines, -len(candidate))


def _is_parseable_python(candidate):
    if not candidate.strip():
        return False

    try:
        ast.parse(candidate)
        return True
    except SyntaxError:
        return False


def _collect_code_candidates(text):
    token_normalized = _decode_generated_text(text)
    token_normalized = token_normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _normalize_generated_text(text)
    candidates = []

    for pattern in (FENCED_CODE_PATTERN, JUPYTER_CODE_PATTERN):
        for match in pattern.findall(token_normalized):
            cleaned = _clean_code_candidate(match)
            if cleaned:
                candidates.append(cleaned)
                expanded = _expand_compact_python(cleaned)
                if expanded != cleaned:
                    candidates.append(expanded)

    cleaned_full_text = _clean_code_candidate(normalized)
    if cleaned_full_text:
        candidates.append(cleaned_full_text)
        expanded_full_text = _expand_compact_python(cleaned_full_text)
        if expanded_full_text != cleaned_full_text:
            candidates.append(expanded_full_text)

        blocks = [block.strip() for block in re.split(r"\n\s*\n+", cleaned_full_text) if block.strip()]
        for block in blocks:
            if STRUCTURED_LINE_PATTERN.match(block) or _is_parseable_python(block):
                candidates.append(block)

        cleaned_lines = cleaned_full_text.splitlines()
        for index, line in enumerate(cleaned_lines):
            if STRUCTURED_LINE_PATTERN.match(line):
                candidate = "\n".join(cleaned_lines[index:]).strip()
                if candidate:
                    candidates.append(candidate)

    deduped_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    return deduped_candidates


def _select_best_candidate(candidates):
    if not candidates:
        return ""

    parseable_python = [candidate for candidate in candidates if _is_parseable_python(candidate)]
    if parseable_python:
        python_like_parseable = [candidate for candidate in parseable_python if _looks_like_python(candidate)]
        ranked = python_like_parseable or parseable_python
        return max(ranked, key=_python_candidate_score).strip()

    python_like = [candidate for candidate in candidates if _looks_like_python(candidate)]
    if python_like:
        return max(python_like, key=_python_candidate_score).strip()

    return max(candidates, key=len).strip()

def extract_python_code(text):
    """
    Extracts purely the python code from the LLM's markdown output.
    """
    candidates = _collect_code_candidates(text)
    if candidates:
        return _select_best_candidate(candidates)

    return _normalize_generated_text(text)


def is_degenerate_output(text, repeated_line_threshold=4, ngram_size=8, repeated_ngram_threshold=3):
    normalized_text = _decode_generated_text(text)
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
