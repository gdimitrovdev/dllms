from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Autoregressive Baseline
ar_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
ar_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct", 
    device_map="auto", 
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
)

def generate_ar(prompt):
    inputs = ar_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = ar_model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0) 
    return ar_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Diffusion LLM
dllm_tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini")
dllm_model = AutoModelForCausalLM.from_pretrained(
    "inclusionAI/LLaDA2.1-mini", 
    trust_remote_code=True, 
    device_map="auto"
)

def generate_dllm(prompt):
    inputs = dllm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = dllm_model.generate(**inputs, steps=64, do_sample=False)
    return dllm_tokenizer.decode(outputs[0], skip_special_tokens=True)
