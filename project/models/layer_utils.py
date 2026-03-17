"""
Layer type mapping utilities.
Inspects model architecture and maps each layer index to its type:
  - "attention"  : full self-attention
  - "linear"     : SSM-based (Mamba-2, GDN, DeltaNet, etc.)
  - "ffn"        : feed-forward only (rare in these hybrids)

Usage:
    python models/layer_utils.py --model qwen3.5-4b
    → prints layer map and saves to models/layer_type_maps/qwen3.5-4b.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.load_model import MODEL_REGISTRY, load_model

# Module class name substrings → layer type
_ATTENTION_KEYWORDS = ["attention", "attn", "selfatt", "mha"]
_LINEAR_KEYWORDS = ["mamba", "gdn", "deltanet", "ssm", "selective", "retention"]

LAYER_TYPE_MAPS_DIR = os.path.join(os.path.dirname(__file__), "layer_type_maps")


def _classify_module(name: str, module_type: str) -> str | None:
    """
    Return 'attention' or 'linear' if the module name/type indicates a sequence
    mixer, else None.
    """
    mt = module_type.lower()
    n = name.lower()
    for kw in _ATTENTION_KEYWORDS:
        if kw in mt or kw in n:
            return "attention"
    for kw in _LINEAR_KEYWORDS:
        if kw in mt or kw in n:
            return "linear"
    return None


def inspect_layer_types(model) -> dict:
    """
    Walk the model's named_modules to find per-layer types.

    Returns:
        {
          "layer_count": N,
          "layers": [
            {"idx": 0, "type": "linear", "module_name": "...", "module_type": "..."},
            ...
          ],
          "attention_indices": [...],
          "linear_indices": [...],
        }
    """
    # Try to locate top-level layer list
    layers = None
    for attr in ["model.layers", "layers", "transformer.h"]:
        try:
            obj = model
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            pass

    if layers is None:
        raise RuntimeError(
            "Could not locate layer list. "
            "Try inspecting model.named_modules() manually."
        )

    layer_info = []
    for idx, layer in enumerate(layers):
        layer_type = "unknown"
        matched_name = ""
        matched_type = ""
        for name, module in layer.named_modules():
            if name == "":
                continue  # skip the layer itself
            mt = type(module).__name__
            result = _classify_module(name, mt)
            if result is not None:
                layer_type = result
                matched_name = name
                matched_type = mt
                break
        layer_info.append(
            {
                "idx": idx,
                "type": layer_type,
                "module_name": matched_name,
                "module_type": matched_type,
            }
        )

    attention_indices = [l["idx"] for l in layer_info if l["type"] == "attention"]
    linear_indices = [l["idx"] for l in layer_info if l["type"] == "linear"]
    unknown_indices = [l["idx"] for l in layer_info if l["type"] == "unknown"]

    result = {
        "layer_count": len(layer_info),
        "layers": layer_info,
        "attention_indices": attention_indices,
        "linear_indices": linear_indices,
        "unknown_indices": unknown_indices,
    }
    return result


def get_layer_type_map(model_name: str, model=None) -> dict:
    """
    Load cached layer type map from disk, or compute and cache it.

    Args:
        model_name: key from MODEL_REGISTRY
        model: if provided, compute map; if None, try loading from cache

    Returns:
        layer type map dict (same as inspect_layer_types output)
    """
    os.makedirs(LAYER_TYPE_MAPS_DIR, exist_ok=True)
    cache_path = os.path.join(LAYER_TYPE_MAPS_DIR, f"{model_name}.json")

    if os.path.exists(cache_path) and model is None:
        with open(cache_path) as f:
            return json.load(f)

    if model is None:
        raise ValueError(
            f"No cached map for '{model_name}' and no model provided. "
            "Pass model= or run layer_utils.py --model <name> first."
        )

    layer_map = inspect_layer_types(model)
    with open(cache_path, "w") as f:
        json.dump(layer_map, f, indent=2)
    print(f"[layer_utils] Saved layer map to {cache_path}")
    return layer_map


def print_layer_summary(layer_map: dict, model_name: str = ""):
    print(f"\n{'='*60}")
    if model_name:
        print(f"  Layer map: {model_name}")
    print(f"  Total layers : {layer_map['layer_count']}")
    print(f"  Attention    : {len(layer_map['attention_indices'])} layers → {layer_map['attention_indices']}")
    print(f"  Linear (SSM) : {len(layer_map['linear_indices'])} layers")
    print(f"  Unknown      : {layer_map['unknown_indices']}")
    print(f"{'='*60}\n")
    print(f"  Layer-by-layer:")
    for l in layer_map["layers"]:
        marker = " <-- ATTN" if l["type"] == "attention" else ""
        print(f"    [{l['idx']:3d}] {l['type']:10s}  ({l['module_name']} : {l['module_type']}){marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and cache layer type map")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device_map", default="cpu", help="Use 'cpu' for quick inspection")
    args = parser.parse_args()

    model, _ = load_model(args.model, device_map=args.device_map)
    layer_map = get_layer_type_map(args.model, model=model)
    print_layer_summary(layer_map, model_name=args.model)
