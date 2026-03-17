"""
Statistical analysis of flip points across models.

Computes:
  1. Chi-square test: are flip points distributed differently between
     attention vs linear layers for hybrid vs TF models?
  2. Flip point depth: normalized layer index at which first flip occurs
     (early = model commits to answer early in the network)
  3. Cross-model comparison table

Usage:
    python analysis/flip_point_stats.py \
        --inputs results/logit_lens_qwen3.5-4b.json results/logit_lens_olmo-3-7b.json \
        --out results/flip_point_stats.json \
        [--plot --out_dir results/figures/]
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_flip_stats(data: dict) -> dict:
    """Extract per-item flip point statistics from logit lens results."""
    model_name = data.get("model", "unknown")
    layer_map = data.get("layer_map", {})
    per_item = data.get("per_item", [])
    n_layers = layer_map.get("layer_count", 1)
    attn_set = set(layer_map.get("attention_indices", []))
    linear_set = set(layer_map.get("linear_indices", []))

    first_flips = []
    first_flip_types = []
    flip_counts = []

    for item in per_item:
        flips = item.get("flip_points", [])
        flip_counts.append(len(flips))

        if flips:
            first = flips[0]
            first_flips.append(first / max(n_layers - 1, 1))  # normalized depth
            if first in attn_set:
                first_flip_types.append("attention")
            elif first in linear_set:
                first_flip_types.append("linear")
            else:
                first_flip_types.append("unknown")
        else:
            first_flips.append(None)
            first_flip_types.append(None)

    n_with_flip = sum(1 for f in first_flips if f is not None)
    depths = [f for f in first_flips if f is not None]
    type_counts = {
        "attention": first_flip_types.count("attention"),
        "linear": first_flip_types.count("linear"),
        "unknown": first_flip_types.count("unknown"),
        "no_flip": first_flip_types.count(None),
    }

    return {
        "model": model_name,
        "n_items": len(per_item),
        "n_with_flip": n_with_flip,
        "pct_with_flip": n_with_flip / len(per_item) if per_item else 0,
        "mean_first_flip_depth": float(np.mean(depths)) if depths else float("nan"),
        "std_first_flip_depth": float(np.std(depths)) if depths else float("nan"),
        "mean_flip_count": float(np.mean(flip_counts)) if flip_counts else float("nan"),
        "first_flip_type_counts": type_counts,
        "pct_first_flip_at_attention": type_counts["attention"] / max(n_with_flip, 1),
        "pct_first_flip_at_linear": type_counts["linear"] / max(n_with_flip, 1),
    }


def chi_square_attn_vs_linear(stats_a: dict, stats_b: dict) -> dict:
    """
    Test whether the distribution of flip points (attention vs linear)
    differs between two models.
    """
    counts_a = stats_a["first_flip_type_counts"]
    counts_b = stats_b["first_flip_type_counts"]

    obs = np.array([
        [counts_a["attention"], counts_a["linear"]],
        [counts_b["attention"], counts_b["linear"]],
    ])

    if not _SCIPY:
        return {"error": "scipy not available", "observed": obs.tolist()}

    if obs.sum() == 0 or (obs < 5).any():
        return {
            "warning": "Low cell counts — chi-square unreliable",
            "observed": obs.tolist(),
        }

    chi2, p, dof, expected = scipy_stats.chi2_contingency(obs)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "observed": obs.tolist(),
        "expected": expected.tolist(),
        "significant_at_0.05": p < 0.05,
    }


def mean_flip_depth_ttest(data_a: dict, data_b: dict) -> dict:
    """T-test: do models differ in where (depth) their first flip occurs?"""
    def _depths(data):
        per_item = data.get("per_item", [])
        n_layers = data.get("layer_map", {}).get("layer_count", 1)
        return [
            fp[0] / max(n_layers - 1, 1)
            for item in per_item
            for fp in [item.get("flip_points", [])]
            if fp
        ]

    d_a = _depths(data_a)
    d_b = _depths(data_b)

    if not d_a or not d_b:
        return {"error": "Insufficient data"}

    result = {
        "model_a": data_a.get("model"),
        "model_b": data_b.get("model"),
        "mean_depth_a": float(np.mean(d_a)),
        "mean_depth_b": float(np.mean(d_b)),
        "n_a": len(d_a),
        "n_b": len(d_b),
    }

    if _SCIPY:
        t, p = scipy_stats.ttest_ind(d_a, d_b)
        result["t_stat"] = float(t)
        result["p_value"] = float(p)
        result["significant_at_0.05"] = p < 0.05

    return result


def print_comparison_table(all_stats: list[dict]):
    print(f"\n{'='*75}")
    print(f"  {'Model':<25} {'%flip':>6} {'depth':>6} {'attn%':>6} {'lin%':>6} {'nflip':>6}")
    print(f"  {'-'*65}")
    for s in all_stats:
        print(
            f"  {s['model']:<25} "
            f"{s['pct_with_flip']:>6.2%} "
            f"{s['mean_first_flip_depth']:>6.3f} "
            f"{s['pct_first_flip_at_attention']:>6.2%} "
            f"{s['pct_first_flip_at_linear']:>6.2%} "
            f"{s['n_with_flip']:>6d}"
        )
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more logit lens JSON result files")
    parser.add_argument("--out", default="results/flip_point_stats.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out_dir", default="results/figures/")
    args = parser.parse_args()

    all_data = [load_json(p) for p in args.inputs]
    all_stats = [extract_flip_stats(d) for d in all_data]

    print_comparison_table(all_stats)

    output = {"per_model": all_stats, "comparisons": []}

    # Pairwise comparisons
    for i in range(len(all_data)):
        for j in range(i + 1, len(all_data)):
            chi2_result = chi_square_attn_vs_linear(all_stats[i], all_stats[j])
            ttest_result = mean_flip_depth_ttest(all_data[i], all_data[j])
            comp = {
                "model_a": all_stats[i]["model"],
                "model_b": all_stats[j]["model"],
                "chi2_attn_vs_linear": chi2_result,
                "ttest_flip_depth": ttest_result,
            }
            output["comparisons"].append(comp)
            print(f"\n  Comparison: {all_stats[i]['model']} vs {all_stats[j]['model']}")
            if "p_value" in chi2_result:
                print(f"    Chi2 (attn/linear dist): p={chi2_result['p_value']:.4f} "
                      f"({'*sig*' if chi2_result.get('significant_at_0.05') else 'n.s.'})")
            if "p_value" in ttest_result:
                print(f"    T-test (flip depth):     p={ttest_result['p_value']:.4f} "
                      f"({'*sig*' if ttest_result.get('significant_at_0.05') else 'n.s.'})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            os.makedirs(args.out_dir, exist_ok=True)

            models = [s["model"] for s in all_stats]
            depths = [s["mean_first_flip_depth"] for s in all_stats]

            fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
            colors = ["steelblue" if "hybrid" in m or "qwen" in m or "nemotron" in m else "salmon"
                      for m in models]
            bars = ax.bar(models, depths, color=colors, edgecolor="black")
            ax.set_ylabel("Mean normalized flip depth", fontsize=11)
            ax.set_title("First Flip Point Depth by Model", fontsize=12)
            ax.set_ylim(0, 1)
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            out_path = os.path.join(args.out_dir, "flip_depth_comparison.pdf")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[plot] Saved → {out_path}")
        except ImportError:
            print("[plot] matplotlib/seaborn not available, skipping plot")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[flip_point_stats] Saved → {args.out}")


if __name__ == "__main__":
    main()
