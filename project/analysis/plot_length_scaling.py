"""
Visualization: Context length vs context_following_rate for all models.

Generates:
  1. Line plot: CFR vs length per model (one line per model, grouped by position)
  2. Delta plot: (hybrid CFR - TF CFR) vs length to highlight divergence

Usage:
    python analysis/plot_length_scaling.py \
        --input results/length_scaling.json \
        --out_dir results/figures/ \
        [--hybrid_models qwen3.5-4b nemotron-h-4b olmo-hybrid-7b] \
        [--tf_models olmo-3-7b llama-3.2-3b]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LENGTH_ORDER = ["500", "2k", "8k", "32k"]
LENGTH_NUMERIC = {"500": 500, "2k": 2000, "8k": 8000, "32k": 32000}
POSITIONS = ["beginning", "middle", "end"]

MODEL_COLORS = {
    "qwen3.5-4b": "#e74c3c",
    "qwen3.5-2b": "#e67e22",
    "nemotron-h-4b": "#9b59b6",
    "nemotron-h-8b": "#8e44ad",
    "olmo-hybrid-7b": "#2980b9",
    "olmo-3-7b": "#27ae60",
    "llama-3.2-3b": "#16a085",
    "gemma-3-4b": "#2ecc71",
}

MODEL_LINESTYLE = {
    "hybrid": "-",
    "transformer": "--",
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, conditions in results.items():
        for key, cond in conditions.items():
            rows.append({
                "model": model_name,
                "length_tag": cond["length"],
                "length_numeric": LENGTH_NUMERIC.get(cond["length"], 0),
                "position": cond["position"],
                "cfr": cond["context_following_rate"],
                "avg_logit_gap": cond["avg_logit_gap"],
                "n": cond["n"],
            })
    return pd.DataFrame(rows)


def plot_cfr_by_length(
    df: pd.DataFrame,
    hybrid_models: list[str],
    tf_models: list[str],
    position: str,
    out_dir: str,
):
    """Line plot of CFR vs length for one position."""
    sub = df[df["position"] == position].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    all_models = hybrid_models + tf_models
    for model_name in all_models:
        m_sub = sub[sub["model"] == model_name].sort_values("length_numeric")
        if m_sub.empty:
            continue
        is_hybrid = model_name in hybrid_models
        color = MODEL_COLORS.get(model_name, "gray")
        ls = "-" if is_hybrid else "--"
        marker = "o" if is_hybrid else "s"
        label = f"{model_name} ({'hybrid' if is_hybrid else 'TF'})"
        ax.plot(
            m_sub["length_numeric"],
            m_sub["cfr"],
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=7,
            linewidth=2,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 8000, 32000])
    ax.set_xticklabels(["500", "2K", "8K", "32K"])
    ax.set_xlabel("Context length (tokens)", fontsize=12)
    ax.set_ylabel("Context-following rate (CFR)", fontsize=12)
    ax.set_title(f"CFR vs Context Length — position={position}", fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cfr_vs_length_{position}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def plot_cfr_all_positions(
    df: pd.DataFrame,
    hybrid_models: list[str],
    tf_models: list[str],
    out_dir: str,
):
    """3-panel plot: one panel per position, all models."""
    positions = [p for p in POSITIONS if p in df["position"].unique()]
    fig, axes = plt.subplots(1, len(positions), figsize=(5 * len(positions), 5), sharey=True)
    if len(positions) == 1:
        axes = [axes]

    all_models = hybrid_models + tf_models
    for ax, position in zip(axes, positions):
        sub = df[df["position"] == position].sort_values("length_numeric")
        for model_name in all_models:
            m_sub = sub[sub["model"] == model_name]
            if m_sub.empty:
                continue
            is_hybrid = model_name in hybrid_models
            color = MODEL_COLORS.get(model_name, "gray")
            ls = "-" if is_hybrid else "--"
            marker = "o" if is_hybrid else "s"
            ax.plot(
                m_sub["length_numeric"],
                m_sub["cfr"],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=6,
                linewidth=1.8,
                label=model_name,
            )
        ax.set_xscale("log")
        ax.set_xticks([500, 2000, 8000, 32000])
        ax.set_xticklabels(["500", "2K", "8K", "32K"], fontsize=9)
        ax.set_title(f"position={position}", fontsize=11)
        ax.set_xlabel("Context length", fontsize=10)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Context-following rate", fontsize=11)
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("CFR vs Context Length by Position", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "cfr_vs_length_all_positions.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def plot_hybrid_tf_delta(
    df: pd.DataFrame,
    hybrid_model: str,
    tf_model: str,
    out_dir: str,
):
    """
    Delta plot: CFR(hybrid) - CFR(TF) vs length.
    Negative delta = hybrid follows context less than TF at that length.
    """
    h = df[df["model"] == hybrid_model]
    t = df[df["model"] == tf_model]
    if h.empty or t.empty:
        return

    merged = h.merge(t, on=["length_numeric", "position"], suffixes=("_h", "_t"))
    merged["delta_cfr"] = merged["cfr_h"] - merged["cfr_t"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for pos in merged["position"].unique():
        sub = merged[merged["position"] == pos].sort_values("length_numeric")
        ax.plot(
            sub["length_numeric"],
            sub["delta_cfr"],
            marker="o",
            linewidth=2,
            label=pos,
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([500, 2000, 8000, 32000])
    ax.set_xticklabels(["500", "2K", "8K", "32K"])
    ax.set_xlabel("Context length (tokens)", fontsize=12)
    ax.set_ylabel("ΔCFR (hybrid − TF)", fontsize=12)
    ax.set_title(f"CFR Gap: {hybrid_model} vs {tf_model}", fontsize=13)
    ax.legend(title="Conflict position", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.fill_between(
        [500, 32000], -1, 0, alpha=0.05, color="red", label="_nolegend_"
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"delta_cfr_{hybrid_model}_vs_{tf_model}.pdf")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to length_scaling.json")
    parser.add_argument("--out_dir", default="results/figures/")
    parser.add_argument("--hybrid_models", nargs="+",
                        default=["qwen3.5-4b", "nemotron-h-4b", "olmo-hybrid-7b"])
    parser.add_argument("--tf_models", nargs="+",
                        default=["olmo-3-7b", "llama-3.2-3b", "gemma-3-4b"])
    args = parser.parse_args()

    data = load_json(args.input)
    df = results_to_dataframe(data)
    print(f"[plot_length_scaling] {len(df)} data points, models: {df['model'].unique().tolist()}")

    os.makedirs(args.out_dir, exist_ok=True)

    for pos in POSITIONS:
        plot_cfr_by_length(df, args.hybrid_models, args.tf_models, pos, args.out_dir)

    plot_cfr_all_positions(df, args.hybrid_models, args.tf_models, args.out_dir)

    # OLMo controlled pair (most important)
    if "olmo-hybrid-7b" in df["model"].values and "olmo-3-7b" in df["model"].values:
        plot_hybrid_tf_delta(df, "olmo-hybrid-7b", "olmo-3-7b", args.out_dir)

    # Any available hybrid vs TF pairs
    available_hybrids = [m for m in args.hybrid_models if m in df["model"].values]
    available_tfs = [m for m in args.tf_models if m in df["model"].values]
    for h in available_hybrids:
        for t in available_tfs:
            if h != "olmo-hybrid-7b" or t != "olmo-3-7b":  # already done above
                plot_hybrid_tf_delta(df, h, t, args.out_dir)


if __name__ == "__main__":
    main()
