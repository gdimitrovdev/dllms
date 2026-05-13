def generate_positional_prompts(before_code, instruction):
    prefix_prompt = f"Instruction: {instruction}\n\nCode to edit:\n```python\n{before_code}\n```\n\nOutput the modified code only."
    suffix_prompt = f"Code to edit:\n```python\n{before_code}\n```\n\nInstruction: {instruction}\n\nOutput the modified code only."

    return prefix_prompt, suffix_prompt

def generate_shuffled_prompts(before_code, constraints_list):
    prompt_ab = f"Instructions:\n1. {constraints_list[0]}\n2. {constraints_list[1]}\n\nCode:\n```python\n{before_code}\n```"
    prompt_ba = f"Instructions:\n1. {constraints_list[1]}\n2. {constraints_list[0]}\n\nCode:\n```python\n{before_code}\n```"

    return prompt_ab, prompt_ba
