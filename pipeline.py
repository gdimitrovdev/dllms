from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc


MAX_GENERATION_TOKENS = 1024
LLADA_THRESHOLD = 0.7
LLADA_EDITING_THRESHOLD = 0.5
LLADA_MAX_POST_STEPS = 16


def _create_bidirectional_mask_shim(config, inputs_embeds, attention_mask=None,
                                    encoder_hidden_states=None, past_key_values=None,
                                    or_mask_function=None, and_mask_function=None, **kwargs):
    """
    Full-attention (bidirectional) mask shim for transformers versions that moved/renamed
    create_bidirectional_mask. Returns None, which tells SDPA to do full attention — correct
    for a bidirectional diffusion LM like LLaDA.
    """
    return None

# Try to import the real one first; fall back to the shim if missing or broken
try:
    import transformers.masking_utils
except (ImportError, ModuleNotFoundError):
    mod = types.ModuleType('transformers.masking_utils')
    sys.modules['transformers.masking_utils'] = mod
    setattr(transformers, 'masking_utils', mod)

transformers.masking_utils.create_bidirectional_mask = _create_bidirectional_mask_shim
print("[Shim] Injected create_bidirectional_mask shim into transformers.masking_utils")


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
        _dllm_tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)
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

def generate_ar(prompt, max_new_tokens=MAX_GENERATION_TOKENS):
    load_ar()

    messages = [
        {"role": "user", "content": prompt}
    ]
    text_input = _ar_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _ar_tokenizer(text_input, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = _ar_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0) 
    generated_tokens = outputs[0][input_length:]
    return _ar_tokenizer.decode(generated_tokens, skip_special_tokens=True)

def generate_dllm(prompt, gen_length=MAX_GENERATION_TOKENS):
    load_dllm()
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = _dllm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)
    outputs = _dllm_model.generate(
        inputs=input_ids,
        gen_length=gen_length,
        block_length=32,
        threshold=LLADA_THRESHOLD,
        editing_threshold=LLADA_EDITING_THRESHOLD,
        max_post_steps=LLADA_MAX_POST_STEPS,
        eos_early_stop=True,
        temperature=0.0,
        top_p=None,
        top_k=None
    )
    return _dllm_tokenizer.decode(outputs[0], skip_special_tokens=True)
