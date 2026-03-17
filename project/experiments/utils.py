"""
Shared utilities for Stage 1 & 2 experiments:
  - load_jsonl / save_jsonl
  - score_response: classify model output as parametric / incontext / other
  - verify_parametric_knowledge: sanity check before conflict experiments
  - generate_answer: run model.generate on a prompt
"""

import json
import os
from typing import Optional

import torch


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def score_response(
    response: str,
    label_parametric: str,
    label_incontext: str,
) -> str:
    """
    Classify model response as:
      "incontext"   → context-following (conflict answer)
      "parametric"  → memory-following (true stored answer)
      "other"       → neither detected
    """
    resp = response.lower().strip()
    lp = label_parametric.lower()
    li = label_incontext.lower()

    # Check multi-word overlap (handles aliases and partial matches)
    def _match(label: str, text: str) -> bool:
        # Direct substring check
        if label in text:
            return True
        # Check any word ≥4 chars in label appears in text
        words = [w for w in label.split() if len(w) >= 4]
        return bool(words) and all(w in text for w in words)

    in_context_match = _match(li, resp)
    parametric_match = _match(lp, resp)

    if in_context_match and not parametric_match:
        return "incontext"
    if parametric_match and not in_context_match:
        return "parametric"
    if in_context_match and parametric_match:
        # Whichever appears first in the response (use safe find to avoid ValueError)
        pos_i = resp.find(li[:4]) if len(li) >= 4 else resp.find(li)
        pos_p = resp.find(lp[:4]) if len(lp) >= 4 else resp.find(lp)
        if pos_i == -1 and pos_p == -1:
            return "other"
        if pos_i == -1:
            return "parametric"
        if pos_p == -1:
            return "incontext"
        return "incontext" if pos_i < pos_p else "parametric"
    return "other"


def compute_logit_gap(
    model,
    tokenizer,
    prompt: str,
    label_parametric: str,
    label_incontext: str,
) -> dict:
    """
    Compute logit gap = logit(incontext_token) - logit(parametric_token)
    at the next-token prediction position.

    Returns dict with keys: logit_gap, logit_parametric, logit_incontext,
    prob_parametric, prob_incontext.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # (vocab,)
    probs = torch.softmax(logits, dim=-1)

    def _get_token_id(label: str) -> int:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        return ids[0] if ids else tokenizer.unk_token_id

    tok_p = _get_token_id(label_parametric)
    tok_i = _get_token_id(label_incontext)

    return {
        "logit_parametric": logits[tok_p].item(),
        "logit_incontext": logits[tok_i].item(),
        "logit_gap": (logits[tok_i] - logits[tok_p]).item(),
        "prob_parametric": probs[tok_p].item(),
        "prob_incontext": probs[tok_i].item(),
        "token_id_parametric": tok_p,
        "token_id_incontext": tok_i,
    }


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Greedy decode up to max_new_tokens, return decoded new tokens only."""
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def verify_parametric_knowledge(
    model,
    tokenizer,
    subject: str,
    true_obj: str,
) -> bool:
    """
    Sanity check: does the model actually know true_obj without any context?
    Only include conflict items where this returns True.
    """
    prompt = f"Q: What is {subject} associated with?\nA:"
    response = generate_answer(model, tokenizer, prompt, max_new_tokens=15)
    return true_obj.lower()[:6] in response.lower()
