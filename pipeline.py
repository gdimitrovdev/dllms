import gc
import sys
import types

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MAX_GENERATION_TOKENS = 512
LLADA21_THRESHOLD = 0.7
LLADA21_EDITING_THRESHOLD = 0.5
LLADA21_MAX_POST_STEPS = 16
LLADA8B_MASK_ID = 126336
LLADA8B_BLOCK_LENGTH = 32
DIFFUCODER_TEMPERATURE = 0.4
DIFFUCODER_TOP_P = 0.95


MODEL_REGISTRY = {
    "qwen25_coder": {
        "label": "Qwen2.5-Coder-7B-Instruct",
        "group": "ar",
        "family": "chat_causal_lm",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "trust_remote_code": False,
        "quantized": True,
        "chat_template_kwargs": {},
        "generation_kwargs": {
            "do_sample": False,
        },
    },
    "deepseek_coder": {
        "label": "DeepSeek-Coder-6.7B-Instruct",
        "group": "ar",
        "family": "chat_causal_lm",
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "trust_remote_code": True,
        "quantized": True,
        "chat_template_kwargs": {},
        "generation_kwargs": {
            "do_sample": False,
        },
    },
    "qwen3_8b": {
        "label": "Qwen3-8B",
        "group": "ar",
        "family": "chat_causal_lm",
        "model_id": "Qwen/Qwen3-8B",
        "trust_remote_code": False,
        "quantized": True,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "generation_kwargs": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        },
    },
    "llada21_mini": {
        "label": "LLaDA2.1-mini",
        "group": "diffusion",
        "family": "llada21",
        "model_id": "inclusionAI/LLaDA2.1-mini",
        "trust_remote_code": True,
        "quantized": True,
        "chat_template_kwargs": {},
    },
    "llada_8b_instruct": {
        "label": "LLaDA-8B-Instruct",
        "group": "diffusion",
        "family": "llada8b",
        "model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "trust_remote_code": True,
        "quantized": True,
        "chat_template_kwargs": {},
    },
    "diffucoder_7b": {
        "label": "DiffuCoder-7B-cpGRPO",
        "group": "diffusion",
        "family": "diffucoder",
        "model_id": "apple/DiffuCoder-7B-cpGRPO",
        "trust_remote_code": True,
        "quantized": True,
        "chat_template_kwargs": {},
    },
}

MODEL_GROUPS = {
    "ar": ["qwen25_coder", "deepseek_coder", "qwen3_8b"],
    "diffusion": ["llada21_mini", "llada_8b_instruct", "diffucoder_7b"],
}


def _create_bidirectional_mask_shim(config, inputs_embeds, attention_mask=None,
                                    encoder_hidden_states=None, past_key_values=None,
                                    or_mask_function=None, and_mask_function=None, **kwargs):
    return None


try:
    import transformers.masking_utils
except (ImportError, ModuleNotFoundError):
    mod = types.ModuleType("transformers.masking_utils")
    sys.modules["transformers.masking_utils"] = mod
    setattr(transformers, "masking_utils", mod)

transformers.masking_utils.create_bidirectional_mask = _create_bidirectional_mask_shim
print("[Shim] Injected create_bidirectional_mask shim into transformers.masking_utils")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def _get_dense_dtype():
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


quantization_config = None
if device == "cuda":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


_loaded_model_key = None
_loaded_model = None
_loaded_tokenizer = None


def _model_load_class(spec):
    if spec["family"] in {"llada8b", "diffucoder"}:
        return AutoModel
    return AutoModelForCausalLM


def _build_model_kwargs(spec, use_quantization):
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": _get_dense_dtype(),
    }
    if spec["trust_remote_code"]:
        model_kwargs["trust_remote_code"] = True
    if use_quantization and quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    return model_kwargs


def _load_model(model_key):
    global _loaded_model_key, _loaded_model, _loaded_tokenizer

    if _loaded_model_key == model_key and _loaded_model is not None and _loaded_tokenizer is not None:
        return _loaded_model, _loaded_tokenizer

    unload_all_models()

    spec = MODEL_REGISTRY[model_key]
    load_class = _model_load_class(spec)
    print(f"Loading {spec['label']}...")

    tokenizer_kwargs = {}
    if spec["trust_remote_code"]:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **tokenizer_kwargs)

    model = None
    wants_quantization = spec["quantized"] and quantization_config is not None
    if wants_quantization:
        try:
            model = load_class.from_pretrained(spec["model_id"], **_build_model_kwargs(spec, use_quantization=True))
            print(f"Loaded {spec['label']} with 4-bit quantization.")
        except Exception as error:
            print(f"4-bit load failed for {spec['label']}: {error}")
            print(f"Falling back to {_get_dense_dtype()} weights for {spec['label']}.")

    if model is None:
        model = load_class.from_pretrained(spec["model_id"], **_build_model_kwargs(spec, use_quantization=False))

    model.eval()

    if model_key == "llada_8b_instruct" and tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    _loaded_model_key = model_key
    _loaded_model = model
    _loaded_tokenizer = tokenizer
    return _loaded_model, _loaded_tokenizer


def unload_all_models():
    global _loaded_model_key, _loaded_model, _loaded_tokenizer

    if _loaded_model is not None:
        print(f"Unloading {_loaded_model_key}...")
        del _loaded_model
        _loaded_model = None

    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None

    _loaded_model_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_prompt_text(tokenizer, spec, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **spec.get("chat_template_kwargs", {}),
    )


def _encode_prompt(model_key, prompt, add_special_tokens=True):
    model, tokenizer = _load_model(model_key)
    spec = MODEL_REGISTRY[model_key]
    prompt_text = _build_prompt_text(tokenizer, spec, prompt)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    return model, tokenizer, encoded


def _generate_causal_lm(model_key, prompt, max_new_tokens=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompt(model_key, prompt)
    input_length = encoded["input_ids"].shape[1]
    generation_kwargs = dict(MODEL_REGISTRY[model_key].get("generation_kwargs", {}))
    generation_kwargs.update({
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    })

    if model_key == "deepseek_coder":
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    outputs = model.generate(**encoded, **generation_kwargs)
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _generate_llada21(prompt, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompt("llada21_mini", prompt)
    outputs = model.generate(
        inputs=encoded["input_ids"],
        gen_length=gen_length,
        block_length=32,
        threshold=LLADA21_THRESHOLD,
        editing_threshold=LLADA21_EDITING_THRESHOLD,
        max_post_steps=LLADA21_MAX_POST_STEPS,
        eos_early_stop=True,
        temperature=0.0,
        top_p=None,
        top_k=None,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _llada8b_add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)).pow(temperature)
    return logits.exp() / gumbel_noise


def _llada8b_get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = torch.div(mask_num, steps, rounding_mode="floor")
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for batch_index in range(mask_num.size(0)):
        num_transfer_tokens[batch_index, :remainder[batch_index].item()] += 1

    return num_transfer_tokens


@torch.no_grad()
def _llada8b_generate(model, prompt_ids, attention_mask=None, steps=MAX_GENERATION_TOKENS,
                      gen_length=MAX_GENERATION_TOKENS, block_length=LLADA8B_BLOCK_LENGTH,
                      temperature=0.0, remasking="low_confidence"):
    sample = torch.full(
        (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
        LLADA8B_MASK_ID,
        dtype=torch.long,
        device=prompt_ids.device,
    )
    sample[:, :prompt_ids.shape[1]] = prompt_ids.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((prompt_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=prompt_ids.device),
            ],
            dim=-1,
        )

    prompt_index = sample != LLADA8B_MASK_ID
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0 or steps % num_blocks != 0:
        raise ValueError("LLaDA-8B requires gen_length divisible by block_length and steps divisible by block count.")

    block_steps = steps // num_blocks

    for block_index in range(num_blocks):
        block_mask_index = (
            sample[:, prompt_ids.shape[1] + block_index * block_length: prompt_ids.shape[1] + (block_index + 1) * block_length] == LLADA8B_MASK_ID
        )
        num_transfer_tokens = _llada8b_get_num_transfer_tokens(block_mask_index, block_steps)

        for step_index in range(block_steps):
            mask_index = sample == LLADA8B_MASK_ID
            logits = model(sample, attention_mask=attention_mask).logits
            logits_with_noise = _llada8b_add_gumbel_noise(logits, temperature=temperature)
            denoised = torch.argmax(logits_with_noise, dim=-1)

            if remasking != "low_confidence":
                raise NotImplementedError(remasking)

            probabilities = F.softmax(logits, dim=-1)
            confidence = torch.squeeze(torch.gather(probabilities, dim=-1, index=torch.unsqueeze(denoised, -1)), -1)
            confidence[:, prompt_ids.shape[1] + (block_index + 1) * block_length:] = -torch.inf

            denoised = torch.where(mask_index, denoised, sample)
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -torch.inf))

            transfer_index = torch.zeros_like(denoised, dtype=torch.bool)
            for batch_index in range(confidence.shape[0]):
                transfer_count = num_transfer_tokens[batch_index, step_index].item()
                if transfer_count <= 0:
                    continue
                _, selected = torch.topk(confidence[batch_index], k=transfer_count)
                transfer_index[batch_index, selected] = True

            sample[transfer_index] = denoised[transfer_index]

    return sample


def _generate_llada8b(prompt, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompt("llada_8b_instruct", prompt, add_special_tokens=False)
    outputs = _llada8b_generate(
        model,
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        steps=gen_length,
        gen_length=gen_length,
        block_length=LLADA8B_BLOCK_LENGTH,
        temperature=0.0,
    )
    generated_tokens = outputs[:, encoded["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded[0].strip()


def _generate_diffucoder(prompt, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompt("diffucoder_7b", prompt)
    input_length = encoded["input_ids"].shape[1]
    outputs = model.diffusion_generate(
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=True,
        steps=gen_length,
        temperature=DIFFUCODER_TEMPERATURE,
        top_p=DIFFUCODER_TOP_P,
        alg="entropy",
        alg_temp=0.0,
    )
    generated_tokens = outputs.sequences[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded.split("<|dlm_pad|>")[0].strip()


def generate_with_model(model_key, prompt, max_new_tokens=MAX_GENERATION_TOKENS):
    family = MODEL_REGISTRY[model_key]["family"]

    if family == "chat_causal_lm":
        return _generate_causal_lm(model_key, prompt, max_new_tokens=max_new_tokens)
    if family == "llada21":
        return _generate_llada21(prompt, gen_length=max_new_tokens)
    if family == "llada8b":
        return _generate_llada8b(prompt, gen_length=max_new_tokens)
    if family == "diffucoder":
        return _generate_diffucoder(prompt, gen_length=max_new_tokens)

    raise ValueError(f"Unsupported model family: {family}")
