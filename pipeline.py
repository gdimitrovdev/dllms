import gc
import sys
import types

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import PreTrainedModel


MAX_GENERATION_TOKENS = 512
LLADA21_THRESHOLD = 0.7
LLADA21_EDITING_THRESHOLD = 0.5
LLADA21_MAX_POST_STEPS = 16
LLADA21_EOS_ID = 156892
LLADA21_MASK_ID = 156895
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
        "batch_size": 8,
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
        "batch_size": 8,
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
        "batch_size": 8,
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
        "batch_size": 2,
        "chat_template_kwargs": {},
    },
    "llada_8b_instruct": {
        "label": "LLaDA-8B-Instruct",
        "group": "diffusion",
        "family": "llada8b",
        "model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "trust_remote_code": True,
        "quantized": False,
        "batch_size": 1,
        "chat_template_kwargs": {},
    },
    "diffucoder_7b": {
        "label": "DiffuCoder-7B-cpGRPO",
        "group": "diffusion",
        "family": "diffucoder",
        "model_id": "apple/DiffuCoder-7B-cpGRPO",
        "trust_remote_code": True,
        "quantized": False,
        "batch_size": 1,
        "chat_template_kwargs": {},
    },
}

MODEL_GROUPS = {
    "ar": ["qwen25_coder", "deepseek_coder", "qwen3_8b"],
    "diffusion": ["llada_8b_instruct", "diffucoder_7b", "llada21_mini"],
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


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = []


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
        "torch_dtype": _get_dense_dtype(),
    }

    if spec["family"] not in {"llada8b", "diffucoder"}:
        model_kwargs["device_map"] = "auto"

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

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

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

    if spec["family"] in {"llada8b", "diffucoder"} and device != "cpu":
        model = model.to(device)

    model.eval()

    if model_key == "llada_8b_instruct" and tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if model_key == "llada_8b_instruct" and tokenizer.pad_token_id == LLADA8B_MASK_ID:
        raise ValueError("LLaDA-8B pad_token_id matches the mask token id; upstream generation code requires a different pad token.")

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
    if spec["family"] == "diffucoder":
        return (
            "<|im_start|>system\n"
            "You are a software engineer. Keep the solutions concise and focused.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt.strip()}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    messages = _build_messages(spec, prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **spec.get("chat_template_kwargs", {}),
    )


def _build_messages(spec, prompt):
    if spec["model_id"] == "deepseek-ai/deepseek-coder-6.7b-instruct":
        return [
            {
                "role": "system",
                "content": (
                    "Return only the full edited Python program. "
                    "Do not include markdown fences, explanations, notebook tags, or any text before or after the code. "
                    "Start directly with Python code and output the final program only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt.strip()}\n\n"
                    "Respond with Python code only. "
                    "No markdown fences. No explanation. No surrounding prose."
                ),
            },
        ]

    return [{"role": "user", "content": prompt}]


def _encode_prompt(model_key, prompt, add_special_tokens=True):
    model, tokenizer, encoded = _encode_prompts(model_key, [prompt], add_special_tokens=add_special_tokens)
    return model, tokenizer, {name: tensor[:1] for name, tensor in encoded.items()}


def _encode_prompts(model_key, prompts, add_special_tokens=True):
    model, tokenizer = _load_model(model_key)
    spec = MODEL_REGISTRY[model_key]

    if spec["family"] == "llada8b":
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        prompt_text = [
            tokenizer.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
                **spec.get("chat_template_kwargs", {}),
            )
            for message in messages
        ]
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    else:
        prompt_text = [_build_prompt_text(tokenizer, spec, prompt) for prompt in prompts]
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )

    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    return model, tokenizer, encoded


def _generate_causal_lm(model_key, prompt, max_new_tokens=MAX_GENERATION_TOKENS):
    return _generate_causal_lm_batch(model_key, [prompt], max_new_tokens=max_new_tokens)[0]


def _generate_causal_lm_batch(model_key, prompts, max_new_tokens=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompts(model_key, prompts)
    input_length = encoded["input_ids"].shape[1]
    generation_kwargs = dict(MODEL_REGISTRY[model_key].get("generation_kwargs", {}))
    generation_kwargs.update({
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    })

    if model_key == "deepseek_coder":
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    outputs = model.generate(**encoded, **generation_kwargs)
    generated_tokens = outputs[:, input_length:]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def _generate_llada21(prompt, gen_length=MAX_GENERATION_TOKENS):
    return _generate_llada21_batch([prompt], gen_length=gen_length)[0]


@torch.no_grad()
def _generate_llada21_batch(prompts, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompts("llada21_mini", prompts)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    prompt_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    batch_size = input_ids.shape[0]
    max_prompt_length = int(prompt_lengths.max().item())
    block_length = 32
    eos_early_stop = True

    num_blocks = (max_prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(torch.bfloat16)
        .repeat(batch_size, 1, 1, 1)
    )

    position_ids = torch.arange(total_length, device=device).unsqueeze(0).repeat(batch_size, 1)
    sample = torch.full(
        (batch_size, total_length),
        LLADA21_MASK_ID,
        dtype=torch.long,
        device=device,
    )

    for batch_index in range(batch_size):
        valid_tokens = input_ids[batch_index, attention_mask[batch_index].bool()]
        sample[batch_index, :valid_tokens.shape[0]] = valid_tokens

    start_block = int(prompt_lengths.min().item()) // block_length
    prompt_lengths_list = [int(length.item()) for length in prompt_lengths]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for block_index in range(start_block, num_blocks):
        current_window_end = (block_index + 1) * block_length
        cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
        cur_position_ids = position_ids[:, :current_window_end]
        block_start_pos = block_index * block_length
        post_steps = torch.zeros(batch_size, dtype=torch.int64, device=device)
        refine_steps = 0
        max_refine_steps = block_length + LLADA21_MAX_POST_STEPS + 4

        while True:
            refine_steps += 1
            if refine_steps > max_refine_steps:
                print(
                    f"[LLaDA2.1] Reached safety cap for block {block_index + 1}; "
                    "moving on to avoid a stalled batch."
                )
                break

            cur_x = sample[:, :current_window_end]
            block_slice = cur_x[:, -block_length:]
            old_block_tokens = block_slice.clone()
            active_block_mask = block_slice == LLADA21_MASK_ID

            no_active_mask = ~active_block_mask.any(dim=1)
            post_steps[no_active_mask] += 1

            if torch.all(no_active_mask & (post_steps > LLADA21_MAX_POST_STEPS)):
                break

            prompt_end_in_block = torch.clamp(
                prompt_lengths - block_start_pos,
                min=0,
                max=block_length,
            )
            block_positions = torch.arange(block_length, device=device).unsqueeze(0)
            prompt_mask_in_block = block_positions < prompt_end_in_block.unsqueeze(1)

            outputs = model.forward(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                output_attentions=False,
            )

            logits = outputs.logits
            active_logits = logits[:, -block_length:, :]
            x0, x0_p = model._sample_with_temperature_topk_topp(
                active_logits,
                temperature=0.0,
                top_k=None,
                top_p=None,
            )

            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)

            for batch_index in range(batch_size):
                if finished[batch_index]:
                    continue

                row_active_mask = active_block_mask[batch_index]
                row_prompt_mask = prompt_mask_in_block[batch_index]
                row_x0 = x0[batch_index]
                row_probs = x0_p[batch_index]
                row_old_tokens = old_block_tokens[batch_index]

                if row_active_mask.any():
                    mask_confidence = torch.where(
                        row_active_mask,
                        row_probs,
                        torch.full_like(row_probs, float("-inf")),
                    )
                    high_conf_mask = (mask_confidence > LLADA21_THRESHOLD) & row_active_mask
                    num_high_confidence = int(high_conf_mask.sum().item())

                    if num_high_confidence >= 1:
                        mask_transfer_index[batch_index] = high_conf_mask
                    else:
                        num_available = int(row_active_mask.sum().item())
                        if num_available > 0:
                            _, idx = torch.topk(
                                mask_confidence,
                                k=min(1, num_available),
                            )
                            mask_transfer_index[batch_index, idx] = True

                non_mask_positions = ~row_active_mask
                non_prompt_positions = ~row_prompt_mask
                editable_positions = non_mask_positions & non_prompt_positions
                if editable_positions.any():
                    editing_confidence = torch.where(
                        editable_positions,
                        row_probs,
                        torch.full_like(row_probs, float("-inf")),
                    )
                    high_conf_editing = (editing_confidence > LLADA21_EDITING_THRESHOLD) & editable_positions
                    token_changed = row_x0 != row_old_tokens
                    editing_transfer_index[batch_index] = high_conf_editing & token_changed

            final_transfer_index = mask_transfer_index | editing_transfer_index
            if final_transfer_index.any():
                block_slice[final_transfer_index] = x0[final_transfer_index]

            if eos_early_stop:
                for batch_index, prompt_length in enumerate(prompt_lengths_list):
                    if finished[batch_index] or prompt_length >= current_window_end:
                        continue

                    generated_part = sample[batch_index, prompt_length:current_window_end]
                    if (generated_part == LLADA21_MASK_ID).sum() == 0:
                        eos_positions = (generated_part == LLADA21_EOS_ID).nonzero(as_tuple=True)[0]
                        if len(eos_positions) > 0:
                            finished[batch_index] = True

            sample[:, :current_window_end] = cur_x

            if torch.all(no_active_mask & (post_steps > LLADA21_MAX_POST_STEPS)):
                break

    decoded = []
    for batch_index, prompt_length in enumerate(prompt_lengths_list):
        generated_answer = sample[batch_index, : prompt_length + gen_length]
        eos_positions = (generated_answer[prompt_length:] == LLADA21_EOS_ID).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_eos_position = int(eos_positions[0].item())
        else:
            first_eos_position = gen_length

        generated_tokens = generated_answer[
            prompt_length: prompt_length + first_eos_position + 1
        ].tolist()
        decoded.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())

    return decoded


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
    return _generate_llada8b_batch([prompt], gen_length=gen_length)[0]


def _generate_llada8b_batch(prompts, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompts("llada_8b_instruct", prompts, add_special_tokens=False)
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
    return [text.strip() for text in decoded]


def _generate_diffucoder(prompt, gen_length=MAX_GENERATION_TOKENS):
    return _generate_diffucoder_batch([prompt], gen_length=gen_length)[0]


def _generate_diffucoder_batch(prompts, gen_length=MAX_GENERATION_TOKENS):
    model, tokenizer, encoded = _encode_prompts("diffucoder_7b", prompts)
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
    generated_tokens = outputs.sequences[:, input_length:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [text.split("<|dlm_pad|>")[0].strip() for text in decoded]


def get_model_batch_size(model_key):
    return MODEL_REGISTRY[model_key].get("batch_size", 1)


def generate_batch_with_model(model_key, prompts, max_new_tokens=MAX_GENERATION_TOKENS):
    family = MODEL_REGISTRY[model_key]["family"]

    if family == "chat_causal_lm":
        return _generate_causal_lm_batch(model_key, prompts, max_new_tokens=max_new_tokens)
    if family == "llada21":
        return _generate_llada21_batch(prompts, gen_length=max_new_tokens)
    if family == "llada8b":
        return _generate_llada8b_batch(prompts, gen_length=max_new_tokens)
    if family == "diffucoder":
        return _generate_diffucoder_batch(prompts, gen_length=max_new_tokens)

    raise ValueError(f"Unsupported model family: {family}")


def generate_with_model(model_key, prompt, max_new_tokens=MAX_GENERATION_TOKENS):
    return generate_batch_with_model(model_key, [prompt], max_new_tokens=max_new_tokens)[0]
