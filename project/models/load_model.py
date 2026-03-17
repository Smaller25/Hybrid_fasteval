"""
Model loading utilities for hybrid LLM knowledge conflict experiments.
Supports: Qwen3.5-4B, Nemotron-H-4B, OLMo-Hybrid-7B, OLMo-3-7B, Llama-3.2-3B, Gemma-3-4B
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _resolve_device_map(device_map: str):
    """
    Resolve device_map for the current platform.
    On Mac M2 (MPS), 'auto' is unreliable — return explicit MPS mapping instead.
    """
    if device_map != "auto":
        return device_map
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


MODEL_REGISTRY = {
    "qwen3.5-4b": {
        "hf_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_gdn",
        "notes": "GDN 3:1, thinking mode off by default",
    },
    "qwen3.5-4b-base": {
        "hf_id": "Qwen/Qwen3.5-4B-Base",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_gdn",
        "notes": "Base model, preferred for research",
    },
    "qwen3.5-2b": {
        "hf_id": "Qwen/Qwen3.5-2B",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_gdn",
        "notes": "Fallback if VRAM insufficient",
    },
    "nemotron-h-4b": {
        "hf_id": "nvidia/Nemotron-H-4B-Instruct-128K",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_mamba2",
        "notes": "Mamba2 dominant, only 4 attn layers, trust_remote_code required",
    },
    "nemotron-h-4b-base": {
        "hf_id": "nvidia/Nemotron-H-4B-Base-8K",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_mamba2",
        "notes": "8K ctx base model",
    },
    "nemotron-h-8b": {
        "hf_id": "nvidia/Nemotron-H-8B-Reasoning-128K",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_mamba2",
        "notes": "8B scale",
    },
    "olmo-hybrid-7b": {
        "hf_id": "allenai/OLMo-Hybrid-7B",
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "arch_type": "hybrid_gdn",
        "notes": "Paired with OLMo-3-7B for controlled comparison",
    },
    "olmo-3-7b": {
        "hf_id": "allenai/OLMo-3-7B",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "transformer",
        "notes": "Controlled TF baseline, same data as OLMo Hybrid",
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "transformer",
        "notes": "Size-matched TF baseline",
    },
    "gemma-3-4b": {
        "hf_id": "google/gemma-3-4b-it",
        "trust_remote_code": False,
        "dtype": torch.bfloat16,
        "arch_type": "transformer",
        "notes": "Size-matched TF baseline",
    },
}


def load_model(model_name: str, device_map: str = "auto"):
    """
    Load model and tokenizer by short name.

    Args:
        model_name: Key from MODEL_REGISTRY (e.g. 'qwen3.5-4b')
        device_map: HF device_map argument

    Returns:
        (model, tokenizer) tuple
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_name]
    hf_id = cfg["hf_id"]
    print(f"[load_model] Loading {model_name} from {hf_id}")
    print(f"  Notes: {cfg['notes']}")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=cfg["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device_map = _resolve_device_map(device_map)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=cfg["dtype"],
        device_map=resolved_device_map,
        trust_remote_code=cfg["trust_remote_code"],
    )
    model.eval()

    print(f"[load_model] Loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model, tokenizer


def get_model_device(model) -> torch.device:
    """Return the device of the first model parameter."""
    return next(model.parameters()).device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device_map)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"[test] Prompt: {prompt}")
    print(f"[test] Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
