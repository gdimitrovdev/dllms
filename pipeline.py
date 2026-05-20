from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

# Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configure 4-bit quantization if on CUDA
quantization_config = None
if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

# Global references to models and tokenizers to allow lazy-loading and dynamic unloading
_ar_model = None
_ar_tokenizer = None
_dllm_model = None
_dllm_tokenizer = None

def load_ar():
    global _ar_model, _ar_tokenizer
    if _ar_model is None:
        # First unload the other model to free VRAM
        unload_dllm()
        print("Loading Autoregressive model (Qwen2.5-Coder-7B-Instruct) in 4-bit...")
        _ar_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
        _ar_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct", 
            device_map="auto", 
            quantization_config=quantization_config,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )

def unload_ar():
    global _ar_model, _ar_tokenizer
    if _ar_model is not None:
        print("Unloading Autoregressive model...")
        del _ar_model
        del _ar_tokenizer
        _ar_model = None
        _ar_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def load_dllm():
    global _dllm_model, _dllm_tokenizer
    if _dllm_model is None:
        # First unload the other model to free VRAM
        unload_ar()
        print("Loading Diffusion LLM (LLaDA2.1-mini) in 4-bit...")
        _dllm_tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini")
        _dllm_model = AutoModelForCausalLM.from_pretrained(
            "inclusionAI/LLaDA2.1-mini", 
            trust_remote_code=True, 
            device_map="auto",
            # quantization_config=quantization_config,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16
        )

def unload_dllm():
    global _dllm_model, _dllm_tokenizer
    if _dllm_model is not None:
        print("Unloading Diffusion LLM...")
        del _dllm_model
        del _dllm_tokenizer
        _dllm_model = None
        _dllm_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_ar(prompt):
    load_ar()
    inputs = _ar_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = _ar_model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0) 
    return _ar_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_dllm(prompt):
    load_dllm()
    inputs = _dllm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = _dllm_model.generate(
        inputs=inputs["input_ids"],
        gen_length=512,
        block_length=32,
        threshold=0.5,
        editing_threshold=0,
        eos_early_stop=True,
        temperature=0.0
    )
    return _dllm_tokenizer.decode(outputs[0], skip_special_tokens=True)
