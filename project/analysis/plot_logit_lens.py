"""
Visualization: Layer-by-layer p(parametric) vs p(incontext) for logit lens results.

Generates:
  1. Line plot: mean p_parametric and p_incontext per layer, with layer type markers
  2. Heatmap: per-item logit_gap across layers

Usage:
    python analysis/plot_logit_lens.py \
        --input results/logit_lens_qwen3.5-4b.json \
        --layer_types models/layer_type_maps/qwen3.5-4b.json \
        --out_dir results/figures/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_mean_probs(per_item: list[dict], layer_map: dict, model_name: str, out_dir: str):
    """Mean p_parametric and p_incontext per layer with type shading."""
    attn_set = set(layer_map["attention_indices"])
    linear_set = set(layer_map["linear_indices"])
    n_layers = layer_map["layer_count"]

    # Aggregate
    layer_p_parametric: dict[int, list] = {i: [] for i in range(n_layers)}
    layer_p_incontext: dict[int, list] = {i: [] for i in range(n_layers)}

    for item in per_item:
        for lp in item.get("layer_probs", []):
            idx = lp["layer"]
            if idx in layer_p_parametric:
                layer_p_parametric[idx].append(lp["p_parametric"])
                layer_p_incontext[idx].append(lp["p_incontext"])

    layers = sorted(layer_p_parametric.keys())
    mean_p = [np.mean(layer_p_parametric[l]) if layer_p_parametric[l] else np.nan for l in layers]
    mean_i = [np.mean(layer_p_incontext[l]) if layer_p_incontext[l] else np.nan for l in layers]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade by layer type
    for l in layers:
        if l in attn_set:
            ax.axvspan(l - 0.5, l + 0.5, alpha=0.15, color="steelblue", label="_nolegend_")
        elif l in linear_set:
            ax.axvspan(l - 0.5, l + 0.5, alpha=0.08, color="salmon", label="_nolegend_")

    ax.plot(layers, mean_p, color="orange", linewidth=2, label="p(parametric)")
    ax.plot(layers, mean_i, color="steelblue", linewidth=2, linestyle="--", label="p(incontext)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)

    # Legend for shading
    attn_patch = mpatches.Patch(color="steelblue", alpha=0.3, label="Attention layer")
    linear_patch = mpatches.Patch(color="salmon", alpha=0.2, label="Linear (SSM) layer")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [attn_patch, linear_patch], fontsize=9)

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Mean probability", fontsize=11)
    ax.set_title(f"Logit Lens — {model_name}", fontsize=13)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"logit_lens_mean_{model_name}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def plot_logit_gap_heatmap(per_item: list[dict], layer_map: dict, model_name: str, out_dir: str):
    """Heatmap of logit_gap (incontext - parametric) per item × layer."""
    n_layers = layer_map["layer_count"]
    n_items = min(len(per_item), 50)  # cap for readability

    matrix = np.full((n_items, n_layers), np.nan)
    for i, item in enumerate(per_item[:n_items]):
        for lp in item.get("layer_probs", []):
            l = lp["layer"]
            if 0 <= l < n_layers:
                matrix[i, l] = lp.get("logit_gap", np.nan)

    fig, ax = plt.subplots(figsize=(16, max(4, n_items // 4)))
    vmax = np.nanpercentile(np.abs(matrix), 95)
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        xticklabels=5,
        yticklabels=False,
        cbar_kws={"label": "logit_gap (incontext − parametric)"},
    )

    # Mark attention layers with vertical lines
    for l in layer_map["attention_indices"]:
        ax.axvline(l + 0.5, color="gold", linewidth=0.6, alpha=0.8)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel(f"Sample (n={n_items})", fontsize=11)
    ax.set_title(f"Logit Gap Heatmap — {model_name}", fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"logit_lens_heatmap_{model_name}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def plot_flip_point_histogram(data: dict, out_dir: str):
    """Bar chart: number of flip points occurring at attention vs linear layers."""
    s = data.get("summary", {})
    model_name = data.get("model", "model")

    categories = ["Attention", "Linear (SSM)", "Unknown"]
    counts = [
        s.get("flip_in_attention_layers", 0),
        s.get("flip_in_linear_layers", 0),
        s.get("flip_in_unknown_layers", 0),
    ]
    colors = ["steelblue", "salmon", "gray"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of flip points", fontsize=11)
    ax.set_title(f"Flip Point Location — {model_name}", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"flip_point_histogram_{model_name}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to logit lens JSON result")
    parser.add_argument("--out_dir", default="results/figures/")
    args = parser.parse_args()

    data = load_json(args.input)
    model_name = data.get("model", "unknown")
    layer_map = data.get("layer_map", {})
    per_item = data.get("per_item", [])

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[plot_logit_lens] Processing {len(per_item)} items for {model_name}")
    plot_mean_probs(per_item, layer_map, model_name, args.out_dir)
    plot_logit_gap_heatmap(per_item, layer_map, model_name, args.out_dir)
    plot_flip_point_histogram(data, args.out_dir)


if __name__ == "__main__":
    main()
