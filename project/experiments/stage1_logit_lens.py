"""
Stage 1 — Logit Lens Analysis

For each layer l, project the hidden state h_l at the last token position
through the final norm + LM head to obtain a probability distribution.
Track p(parametric_token) and p(incontext_token) across layers.

Key output: "flip_point" — the layer at which the dominant prediction
switches from parametric to incontext (or vice versa), and whether that
flip coincides with an attention or linear layer.

Usage:
    python experiments/stage1_logit_lens.py \
        --model qwen3.5-4b \
        --data data/output/short_conflict.jsonl \
        --n_samples 50 \
        --out results/logit_lens_qwen3.5-4b.json
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.load_model import MODEL_REGISTRY, load_model
from models.layer_utils import get_layer_type_map
from experiments.utils import load_jsonl, save_json


def _get_final_norm(model):
    """Locate the final layer norm before LM head."""
    for attr in ["model.norm", "model.final_layer_norm", "transformer.ln_f", "norm"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            pass
    return None


def _get_lm_head(model):
    for attr in ["lm_head", "embed_out"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Could not locate lm_head on model")


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


def logit_lens_analysis(
    model,
    tokenizer,
    prompt: str,
    label_parametric: str,
    label_incontext: str,
) -> list[dict]:
    """
    Run logit lens for a single prompt.
    Returns list of per-layer dicts with p_parametric, p_incontext.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    hidden_states_by_layer: dict[int, torch.Tensor] = {}

    layers = _get_layers(model)
    hooks = []
    for i, layer in enumerate(layers):
        def _hook(module, inp, out, idx=i):
            hs = out[0] if isinstance(out, tuple) else out
            hidden_states_by_layer[idx] = hs.detach()
        hooks.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    norm = _get_final_norm(model)
    lm_head = _get_lm_head(model)

    def _get_token_id(label: str) -> int:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        return ids[0] if ids else tokenizer.unk_token_id

    tok_p = _get_token_id(label_parametric)
    tok_i = _get_token_id(label_incontext)

    layer_probs = []
    for layer_idx in sorted(hidden_states_by_layer.keys()):
        h = hidden_states_by_layer[layer_idx]  # (batch, seq, hidden)
        h_last = h[0, -1, :]  # last token
        with torch.no_grad():
            if norm is not None:
                h_last = norm(h_last.unsqueeze(0)).squeeze(0)
            logits = lm_head(h_last)
        probs = torch.softmax(logits.float(), dim=-1)

        p_p = probs[tok_p].item()
        p_i = probs[tok_i].item()
        dominant = "parametric" if p_p > p_i else "incontext"

        layer_probs.append({
            "layer": layer_idx,
            "p_parametric": p_p,
            "p_incontext": p_i,
            "logit_gap": (logits[tok_i] - logits[tok_p]).item(),
            "dominant": dominant,
        })

    return layer_probs


def find_flip_points(layer_probs: list[dict]) -> list[int]:
    """Return layer indices where dominant prediction changes."""
    flips = []
    for i in range(1, len(layer_probs)):
        if layer_probs[i]["dominant"] != layer_probs[i - 1]["dominant"]:
            flips.append(layer_probs[i]["layer"])
    return flips


def run_logit_lens(
    model,
    tokenizer,
    records: list[dict],
    model_name: str,
) -> dict:
    layer_map = get_layer_type_map(model_name, model=model)
    attn_set = set(layer_map["attention_indices"])
    linear_set = set(layer_map["linear_indices"])

    all_results = []
    flip_in_attn = 0
    flip_in_linear = 0
    flip_in_unknown = 0

    for rec in tqdm(records, desc="Logit lens"):
        lp = rec["label_parametric"]
        li = rec["label_incontext"]
        prompt = rec["prompt_with_context"]

        layer_probs = logit_lens_analysis(model, tokenizer, prompt, lp, li)
        flips = find_flip_points(layer_probs)

        for f in flips:
            if f in attn_set:
                flip_in_attn += 1
            elif f in linear_set:
                flip_in_linear += 1
            else:
                flip_in_unknown += 1

        # Final layer verdict
        final = layer_probs[-1] if layer_probs else {}

        all_results.append({
            "id": rec.get("id", ""),
            "subject": rec["subject"],
            "label_parametric": lp,
            "label_incontext": li,
            "flip_points": flips,
            "final_dominant": final.get("dominant", "unknown"),
            "final_p_parametric": final.get("p_parametric", 0.0),
            "final_p_incontext": final.get("p_incontext", 0.0),
            "layer_probs": layer_probs,
        })

    total_flips = flip_in_attn + flip_in_linear + flip_in_unknown
    summary = {
        "n_evaluated": len(all_results),
        "flip_in_attention_layers": flip_in_attn,
        "flip_in_linear_layers": flip_in_linear,
        "flip_in_unknown_layers": flip_in_unknown,
        "total_flips": total_flips,
        "pct_flip_at_attention": flip_in_attn / total_flips if total_flips else float("nan"),
        "pct_flip_at_linear": flip_in_linear / total_flips if total_flips else float("nan"),
        "final_dominant_counts": {
            "incontext": sum(1 for r in all_results if r["final_dominant"] == "incontext"),
            "parametric": sum(1 for r in all_results if r["final_dominant"] == "parametric"),
        },
    }

    return {
        "model": model_name,
        "layer_map": layer_map,
        "summary": summary,
        "per_item": all_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data", default="data/output/short_conflict.jsonl")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--out", default="results/logit_lens.json")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device_map=args.device_map)
    records = load_jsonl(args.data)[: args.n_samples]
    print(f"[stage1_logit_lens] Running on {len(records)} samples with {args.model}")

    out = run_logit_lens(model, tokenizer, records, args.model)
    save_json(out, args.out)

    s = out["summary"]
    print(f"\n{'='*55}")
    print(f"  Model: {args.model}")
    print(f"  Flip at attention layers : {s['flip_in_attention_layers']} ({s['pct_flip_at_attention']:.1%})")
    print(f"  Flip at linear layers    : {s['flip_in_linear_layers']} ({s['pct_flip_at_linear']:.1%})")
    print(f"  Final dominant incontext : {s['final_dominant_counts']['incontext']}")
    print(f"  Final dominant parametric: {s['final_dominant_counts']['parametric']}")
    print(f"{'='*55}")
    print(f"[stage1_logit_lens] Saved → {args.out}")


if __name__ == "__main__":
    main()
