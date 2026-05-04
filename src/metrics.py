"""
Evaluation Metrics for LTE Reproducibility
============================================
Implements the four metrics from Table 1 of Jiang et al. (2024):

1. Edit Success  — does the model correctly answer queries about the edited fact?
2. Portability   — does the model apply the edit to logically related queries?
3. Locality      — does the model leave unrelated knowledge unchanged?
4. Fluency       — does the model generate fluent text? (measured by perplexity)
"""

import math
import re
import string
from typing import Dict, List, Optional
import numpy as np


# ── Exact Match Helpers ───────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth) -> bool:
    """Check if ground truth appears anywhere in prediction. Handles list ground_truth."""
    if isinstance(ground_truth, list):
        return any(contains_match(prediction, g) for g in ground_truth)
    return normalize_answer(str(ground_truth)) in normalize_answer(str(prediction))


# ── Edit Success ──────────────────────────────────────────────────────────────

def compute_edit_success(
    predictions: List[str],
    targets: List[str],
    match_fn: str = "contains",
) -> float:
    """
    Edit Success: proportion of edited facts correctly recalled.
    Uses contains-match (more lenient) as in paper.

    Args:
        predictions : model outputs for edited fact queries
        targets     : target_new strings (the new facts)
        match_fn    : 'exact' or 'contains'

    Returns:
        edit_success (float, 0-100)
    """
    assert len(predictions) == len(targets)
    fn = contains_match if match_fn == "contains" else exact_match
    correct = sum(fn(p, t) for p, t in zip(predictions, targets))
    return 100.0 * correct / len(predictions)


# ── Portability ───────────────────────────────────────────────────────────────

def compute_portability(
    predictions: List[str],
    targets: List[str],
    match_fn: str = "contains",
) -> float:
    """
    Portability: ability to apply edit to logically related (indirect) questions.
    Same formula as edit success but on portability queries.

    Returns:
        portability (float, 0-100)
    """
    if not predictions:
        return 0.0
    assert len(predictions) == len(targets)
    fn = contains_match if match_fn == "contains" else exact_match
    correct = sum(fn(p, t) for p, t in zip(predictions, targets))
    return 100.0 * correct / len(predictions)


# ── Locality ──────────────────────────────────────────────────────────────────

def compute_locality(
    pre_edit_predictions: List[str],
    post_edit_predictions: List[str],
) -> float:
    """
    Locality: proportion of out-of-scope queries where the model's answer
    does NOT change after editing (preserving unrelated knowledge).

    Args:
        pre_edit_predictions  : model answers BEFORE editing
        post_edit_predictions : model answers AFTER editing

    Returns:
        locality (float, 0-100)
    """
    assert len(pre_edit_predictions) == len(post_edit_predictions)
    if not pre_edit_predictions:
        return 100.0
    unchanged = sum(
        normalize_answer(pre) == normalize_answer(post)
        for pre, post in zip(pre_edit_predictions, post_edit_predictions)
    )
    return 100.0 * unchanged / len(pre_edit_predictions)


# ── Fluency ───────────────────────────────────────────────────────────────────

def compute_fluency_from_logprobs(logprobs: List[float]) -> float:
    """
    Fluency: measured as weighted perplexity of generated text.
    Lower perplexity = higher fluency.

    Paper uses weighted perplexity (WPL) from MEND / ROME eval suite.
    We convert to a 0-100 score: fluency = 100 * exp(-avg_logprob)

    Args:
        logprobs: list of per-token log probabilities from model

    Returns:
        fluency score (float) — higher is better
    """
    if not logprobs:
        return 0.0
    avg_nll = -np.mean(logprobs)
    perplexity = math.exp(avg_nll)
    # Convert to 0-100 score: higher = more fluent
    # Typical PPL for LLaMA2 on natural text: 5-15
    # Score = max(0, 100 - 5 * log2(PPL))
    score = max(0.0, 100.0 - 5.0 * math.log2(max(perplexity, 1.0)))
    return score


# ── Aggregate Results ─────────────────────────────────────────────────────────

def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate per-sample results into a summary table matching paper Table 1.

    Args:
        results: list of dicts, each with keys:
            edit_success, portability, locality, fluency

    Returns:
        dict with mean values for each metric
    """
    keys = ["edit_success", "portability", "locality", "fluency"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if k in r and r[k] is not None]
        summary[k] = round(float(np.mean(vals)), 2) if vals else None
    return summary


def print_results_table(results: Dict, model_name: str, dataset: str):
    """Print results in a format matching paper Table 1."""
    print(f"\n{'='*60}")
    print(f"  Model  : {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Score':>10}")
    print(f"  {'-'*30}")
    metrics = [
        ("Edit Success",  "edit_success"),
        ("Portability",   "portability"),
        ("Locality",      "locality"),
        ("Fluency",       "fluency"),
    ]
    for label, key in metrics:
        val = results.get(key)
        val_str = f"{val:.2f}" if val is not None else "N/A"
        print(f"  {label:<20} {val_str:>10}")
    print(f"{'='*60}\n")


def compare_with_paper(our_results: Dict, dataset: str):
    """
    Compare our results with paper Table 1 numbers.
    Paper: LTE-LoRA on LLaMA2-Chat-7B (Table 1, Jiang et al. 2024)
    """
    # Numbers from Table 1 of the paper
    paper_numbers = {
        "ZsRE": {
            "LTE-LoRA": {
                "edit_success": 99.9,
                "portability":  57.0,
                "locality":     100.0,
                "fluency":      None,  # paper reports as score
            },
            "ROME": {
                "edit_success": 96.6,
                "portability":  36.1,
                "locality":     53.1,
                "fluency":      None,
            },
            "ICE": {
                "edit_success": 69.3,
                "portability":  47.6,
                "locality":     100.0,
                "fluency":      None,
            },
        },
        "WikiDatacounterfact": {
            "LTE-LoRA": {
                "edit_success": 99.9,
                "portability":  46.1,
                "locality":     100.0,
                "fluency":      None,
            },
            "ROME": {
                "edit_success": 95.2,
                "portability":  33.0,
                "locality":     58.2,
                "fluency":      None,
            },
            "ICE": {
                "edit_success": 59.2,
                "portability":  40.4,
                "locality":     100.0,
                "fluency":      None,
            },
        },
    }

    if dataset not in paper_numbers:
        print(f"No paper numbers available for dataset: {dataset}")
        return

    print(f"\n{'='*70}")
    print(f"  Comparison with Paper Table 1 — {dataset}")
    print(f"{'='*70}")
    print(f"  {'Metric':<18} {'Ours':>10} {'Paper(LTE-LoRA)':>16} {'Delta':>8}")
    print(f"  {'-'*52}")
    paper = paper_numbers[dataset].get("LTE-LoRA", {})
    for metric in ["edit_success", "portability", "locality", "fluency"]:
        ours  = our_results.get(metric)
        theirs = paper.get(metric)
        if ours is not None and theirs is not None:
            delta = ours - theirs
            sign  = "+" if delta >= 0 else ""
            print(f"  {metric:<18} {ours:>10.2f} {theirs:>16.2f} {sign}{delta:>7.2f}")
        else:
            print(f"  {metric:<18} {'N/A':>10} {'N/A':>16} {'N/A':>8}")
    print(f"{'='*70}\n")
