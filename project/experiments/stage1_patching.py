"""
Stage 1 — Activation Patching (Causal Mediation Analysis)

Protocol:
  - "clean" run  : prompt WITHOUT conflict context (no external info)
  - "corrupted" run: prompt WITH conflict context (counterfactual)

For each layer, replace the corrupted run's hidden state at that layer
with the clean run's hidden state, then measure the change in prediction:

  patch_effect(layer) = logit_gap_after_patch - logit_gap_before_patch

A large positive patch_effect at a linear layer means patching in the
clean (parametric) signal at that layer moves the model toward the
parametric answer → evidence that linear layers mediate parametric bias.

Usage:
    python experiments/stage1_patching.py \
        --model qwen3.5-4b \
        --data data/output/short_conflict.jsonl \
        --n_samples 20 \
        --patch_targets attention linear \
        --out results/patching_qwen3.5-4b.json
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.load_model import MODEL_REGISTRY, load_model
from models.layer_utils import get_layer_type_map
from experiments.utils import load_jsonl, save_json


def _get_layers(model):
    for attr in ["model.layers", "layers", "transformer.h"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    raise RuntimeError("Could not locate model layer list")


def _get_lm_head(model):
    for attr in ["lm_head", "embed_out"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Could not locate lm_head")


def _get_final_norm(model):
    for attr in ["model.norm", "model.final_layer_norm", "transformer.ln_f"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    return None


def _collect_hidden_states(model, inputs: dict) -> dict[int, torch.Tensor]:
    """Run forward pass and collect hidden states from all layers."""
    states = {}
    layers = _get_layers(model)
    hooks = []
    for i, layer in enumerate(layers):
        def _hook(module, inp, out, idx=i):
            hs = out[0] if isinstance(out, tuple) else out
            states[idx] = hs.detach().clone()
        hooks.append(layer.register_forward_hook(_hook))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return states


def _compute_logit_gap(
    model,
    inputs: dict,
    patch_layer: int,
    patch_state: torch.Tensor,
    tok_p: int,
    tok_i: int,
) -> float:
    """
    Run forward pass with a patch at patch_layer, return logit_gap.
    If patch_layer == -1, no patch (baseline run).
    """
    norm = _get_final_norm(model)
    lm_head = _get_lm_head(model)

    last_hs = {}
    layers = _get_layers(model)
    hooks = []

    def _collect_last(module, inp, out, idx):
        hs = out[0] if isinstance(out, tuple) else out
        last_hs["val"] = hs.detach()

    # Patch hook: replace output of patch_layer
    def _patch_hook(module, inp, out):
        if isinstance(out, tuple):
            return (patch_state.to(out[0].device),) + out[1:]
        return patch_state.to(out.device)

    for i, layer in enumerate(layers):
        if i == patch_layer:
            hooks.append(layer.register_forward_hook(_patch_hook))
        elif i == len(layers) - 1:
            hooks.append(layer.register_forward_hook(
                lambda m, inp, out, _i=i: last_hs.update({"val": (out[0] if isinstance(out, tuple) else out).detach()})
            ))

    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()

    if "val" not in last_hs:
        return float("nan")

    h_last = last_hs["val"][0, -1, :]
    with torch.no_grad():
        if norm is not None:
            h_last = norm(h_last.unsqueeze(0)).squeeze(0)
        logits = lm_head(h_last).float()
    return (logits[tok_i] - logits[tok_p]).item()


def activation_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    label_parametric: str,
    label_incontext: str,
    layer_indices: list[int],
) -> list[dict]:
    """
    Patch each layer in layer_indices and measure effect on logit_gap.

    Returns list of {layer, baseline_gap, patched_gap, patch_effect}.
    """
    device = next(model.parameters()).device

    def _tokenize(p):
        return tokenizer(p, return_tensors="pt").to(device)

    inputs_clean = _tokenize(clean_prompt)
    inputs_corrupted = _tokenize(corrupted_prompt)

    def _get_token_id(label: str) -> int:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        return ids[0] if ids else tokenizer.unk_token_id

    tok_p = _get_token_id(label_parametric)
    tok_i = _get_token_id(label_incontext)

    # Collect clean hidden states
    clean_states = _collect_hidden_states(model, inputs_clean)

    # Baseline: corrupted run without patch
    baseline_gap = _compute_logit_gap(model, inputs_corrupted, -1, None, tok_p, tok_i)

    results = []
    for layer_idx in tqdm(layer_indices, desc="  Patching layers", leave=False):
        if layer_idx not in clean_states:
            continue
        patch_state = clean_states[layer_idx]
        patched_gap = _compute_logit_gap(
            model, inputs_corrupted, layer_idx, patch_state, tok_p, tok_i
        )
        results.append({
            "layer": layer_idx,
            "baseline_gap": baseline_gap,
            "patched_gap": patched_gap,
            # Positive = patch moved prediction toward parametric
            "patch_effect": baseline_gap - patched_gap,
        })

    return results


def run_patching(
    model,
    tokenizer,
    records: list[dict],
    model_name: str,
    patch_targets: list[str],
) -> dict:
    layer_map = get_layer_type_map(model_name, model=model)

    target_indices = []
    for pt in patch_targets:
        if pt == "attention":
            target_indices.extend(layer_map["attention_indices"])
        elif pt == "linear":
            target_indices.extend(layer_map["linear_indices"])
        elif pt == "all":
            target_indices = list(range(layer_map["layer_count"]))
            break
    target_indices = sorted(set(target_indices))

    all_results = []
    for rec in tqdm(records, desc="Activation patching"):
        layer_results = activation_patch(
            model=model,
            tokenizer=tokenizer,
            clean_prompt=rec["prompt_no_context"],
            corrupted_prompt=rec["prompt_with_context"],
            label_parametric=rec["label_parametric"],
            label_incontext=rec["label_incontext"],
            layer_indices=target_indices,
        )
        all_results.append({
            "id": rec.get("id", ""),
            "subject": rec["subject"],
            "label_parametric": rec["label_parametric"],
            "label_incontext": rec["label_incontext"],
            "layer_patch_results": layer_results,
        })

    # Aggregate: mean patch_effect per layer, by layer type
    from collections import defaultdict
    attn_set = set(layer_map["attention_indices"])
    linear_set = set(layer_map["linear_indices"])

    layer_effects: dict[int, list[float]] = defaultdict(list)
    for item in all_results:
        for lr in item["layer_patch_results"]:
            if lr["patched_gap"] != lr["patched_gap"]:  # nan
                continue
            layer_effects[lr["layer"]].append(lr["patch_effect"])

    layer_summary = []
    for layer_idx in sorted(layer_effects.keys()):
        effects = layer_effects[layer_idx]
        ltype = "attention" if layer_idx in attn_set else ("linear" if layer_idx in linear_set else "unknown")
        layer_summary.append({
            "layer": layer_idx,
            "type": ltype,
            "mean_patch_effect": sum(effects) / len(effects),
            "n": len(effects),
        })

    # Top-k layers by effect
    top_layers = sorted(layer_summary, key=lambda x: abs(x["mean_patch_effect"]), reverse=True)[:10]

    return {
        "model": model_name,
        "patch_targets": patch_targets,
        "layer_map": layer_map,
        "summary": {
            "n_evaluated": len(all_results),
            "top_layers_by_effect": top_layers,
        },
        "layer_effects": layer_summary,
        "per_item": all_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data", default="data/output/short_conflict.jsonl")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--patch_targets", nargs="+",
                        choices=["attention", "linear", "all"],
                        default=["attention", "linear"])
    parser.add_argument("--out", default="results/patching.json")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device_map=args.device_map)
    records = load_jsonl(args.data)[: args.n_samples]
    print(f"[stage1_patching] Running on {len(records)} samples with {args.model}")

    out = run_patching(model, tokenizer, records, args.model, args.patch_targets)
    save_json(out, args.out)

    print(f"\n{'='*55}")
    print(f"  Model: {args.model}")
    print("  Top layers by mean patch effect:")
    for l in out["summary"]["top_layers_by_effect"][:5]:
        print(f"    Layer {l['layer']:3d} ({l['type']:10s}): {l['mean_patch_effect']:+.4f}")
    print(f"{'='*55}")
    print(f"[stage1_patching] Saved → {args.out}")


if __name__ == "__main__":
    main()
