"""
Shared utility functions for Hybrid_fasteval notebooks.

Provides helper functions for:
- Project path setup
- Model information formatting
- Prompt display
- Verdict styling
- Hidden state extraction
- Logit lens computation
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
import torch.nn.functional as F


def setup_project_path(project_root: str = "/Users/smaller225/code/Hybrid_fasteval") -> Path:
    """
    Add project root to sys.path and return Path object.

    Args:
        project_root: Absolute path to project root

    Returns:
        Path object for project root
    """
    project_path = Path(project_root)
    project_module_path = project_path / "project"

    if str(project_module_path) not in sys.path:
        sys.path.insert(0, str(project_module_path))
        print(f"✓ Added {project_module_path} to sys.path")

    return project_path


def format_model_info(model, tokenizer=None) -> pd.DataFrame:
    """
    Format model information as a DataFrame for display.

    Args:
        model: HuggingFace model
        tokenizer: Optional tokenizer

    Returns:
        DataFrame with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "Property": [
            "Total Parameters",
            "Trainable Parameters",
            "Model Type",
            "Device",
            "Dtype",
        ],
        "Value": [
            f"{total_params:,}",
            f"{trainable_params:,}",
            model.config.model_type if hasattr(model, 'config') else "Unknown",
            str(next(model.parameters()).device),
            str(next(model.parameters()).dtype),
        ]
    }

    if tokenizer is not None:
        info["Property"].append("Vocab Size")
        info["Value"].append(f"{len(tokenizer):,}")

    return pd.DataFrame(info)


def display_prompt_preview(prompt: str, title: str = "Prompt", max_chars: int = 500) -> None:
    """
    Display prompt with truncation if too long.

    Args:
        prompt: Prompt text
        title: Title for display
        max_chars: Maximum characters to display
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Length: {len(prompt)} characters\n")

    if len(prompt) <= max_chars:
        print(prompt)
    else:
        half = max_chars // 2
        print(prompt[:half])
        print(f"\n... [{len(prompt) - max_chars} characters omitted] ...\n")
        print(prompt[-half:])

    print(f"{'='*60}\n")


def color_verdict(val: str) -> str:
    """
    Return CSS background color for verdict styling.

    Args:
        val: Verdict value (incontext/parametric/other)

    Returns:
        CSS color string
    """
    if val == 'incontext':
        return 'background-color: lightblue'
    elif val == 'parametric':
        return 'background-color: lightcoral'
    elif val == 'other':
        return 'background-color: lightyellow'
    return ''


def extract_hidden_states(
    model,
    inputs: Dict[str, torch.Tensor],
    layer_module_path: str = "model.layers"
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states from all transformer layers using forward hooks.

    Args:
        model: HuggingFace model
        inputs: Tokenized inputs dict
        layer_module_path: Path to layers (e.g., "model.layers")

    Returns:
        Dict mapping layer index to hidden state tensor (1, seq_len, hidden_dim)
    """
    hidden_states_dict = {}
    hooks = []

    # Get layers module
    layers = model
    for attr in layer_module_path.split('.'):
        layers = getattr(layers, attr)

    # Register hooks
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Output is typically a tuple, first element is hidden states
            if isinstance(output, tuple):
                hidden_states_dict[layer_idx] = output[0].detach().cpu()
            else:
                hidden_states_dict[layer_idx] = output.detach().cpu()
        return hook

    for idx, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_hook(idx))
        hooks.append(hook)

    # Forward pass
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return hidden_states_dict


def compute_logit_lens(
    hidden_states: torch.Tensor,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: str = "cpu",
    top_k: int = 5
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """
    Project hidden state to vocabulary space (logit lens).

    Args:
        hidden_states: Hidden state tensor (1, seq_len, hidden_dim)
        final_norm: Final layer normalization
        lm_head: Language model head
        device: Device for computation
        top_k: Number of top predictions to return

    Returns:
        Tuple of (logits, top_k_token_ids, top_k_probs)
    """
    # Take last token
    last_hidden = hidden_states[:, -1, :].to(device)

    # Apply final norm
    normed = final_norm(last_hidden)

    # Project to vocab
    logits = lm_head(normed)

    # Get top-k
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs[0], k=top_k)

    return logits, top_k_indices.cpu().tolist(), top_k_probs.cpu().tolist()


def get_final_norm(model):
    """Get final normalization layer from model."""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    elif hasattr(model, 'norm'):
        return model.norm
    else:
        raise AttributeError("Could not find final normalization layer")


def get_lm_head(model):
    """Get language model head from model."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    else:
        raise AttributeError("Could not find lm_head")
