"""
ROME Baseline Evaluation
=========================
Evaluates ROME (Rank-One Model Editing) on ZsRE and WikiDatacounterfact.
Imports ROME core modules directly from EasyEdit, bypassing the multimodal
dependency chain in easyeditor/__init__.py.

Run in easyedit-env:
    python scripts/run_rome.py --dataset ZsRE
    python scripts/run_rome.py --dataset WikiDatacounterfact

First run: downloads Wikipedia covariance stats (~400 MB, ~5-10 min).
Cached to ./data/rome_stats/ on subsequent runs.
"""

import os
import sys
import json
import argparse
import re
import string
import copy
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Import ROME core directly (bypasses broken multimodal __init__ chain) ─────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EASY_DIR = os.path.join(ROOT, "EasyEdit")
sys.path.insert(0, ROOT)
sys.path.insert(0, EASY_DIR)

from easyeditor.models.rome.rome_hparams import ROMEHyperParams
from easyeditor.models.rome.rome_main import apply_rome_to_model
from easyeditor.util import nethook

# ── Paths & constants ─────────────────────────────────────────────────────────
ZSRE_PATH   = os.path.join(ROOT, "data/knowedit/benchmark/ZsRE/ZsRE-test-all.json")
WIKICF_PATH = os.path.join(ROOT, "data/knowedit/benchmark/wiki_counterfact/test_cf.json")
HPARAMS     = os.path.join(ROOT, "configs/rome_llama2_hparams.yaml")

MODEL_NAME  = "NousResearch/Llama-2-7b-chat-hf"

PAPER_NUMBERS = {
    "ZsRE":               {"edit_success": 96.6, "portability": 36.1, "locality": 53.1},
    "WikiDatacounterfact": {"edit_success": 99.9, "portability": 35.4, "locality": 29.8},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(s):
    if isinstance(s, list):
        s = s[0] if s else ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def contains_match(pred, target):
    if isinstance(target, list):
        return any(contains_match(pred, t) for t in target)
    return normalize(str(target)) in normalize(str(pred))


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=40, device="cuda"):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(device="cuda"):
    """Load model directly to GPU without device_map auto-offloading.
    ROME uses nethook to hook into model layers — CPU offloading via
    device_map='auto' breaks the hook mechanism. Force all layers to GPU."""
    print(f"Loading {MODEL_NAME} in fp16 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tok


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, tok, hparams, dataset, max_samples, device="cuda"):
    if max_samples:
        dataset = dataset[:max_samples]

    edit_hits = 0
    total = len(dataset)
    port_scores, loc_scores = [], []

    for item in tqdm(dataset, desc="ROME Evaluation"):
        subj  = item.get("subject", "")
        tgt   = item.get("target_new", "")
        query = item.get("prompt", "")

        if isinstance(tgt, list):
            tgt = tgt[0]

        # ── Build ROME request ────────────────────────────────────────────
        request = {
            "prompt":     query,        # e.g. "The capital of France is {}"
            "subject":    subj,         # e.g. "France"
            "target_new": tgt,          # e.g. "Berlin"
        }

        # ── Apply ROME edit (returns edited model + saved weights) ────────
        weights_copy = {}
        try:
            edited_model, weights_copy = apply_rome_to_model(
                model, tok, [request], hparams,
                copy=False,
                return_orig_weights=True,
            )
        except Exception as e:
            print(f"\n[WARN] ROME failed for '{query[:60]}': {e}")
            # Restore any partially-modified weights before skipping
            if weights_copy:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(model, k)[...] = v.to(device)
            continue

        # ── Edit Success ──────────────────────────────────────────────────
        pred = generate(edited_model, tok, query, device=device)
        if contains_match(pred, tgt):
            edit_hits += 1

        # ── Portability ───────────────────────────────────────────────────
        p_hits, p_tot = 0, 0
        for _, port_items in item.get("portability", {}).items():
            if not isinstance(port_items, list):
                continue
            for pi in port_items:
                pq = pi.get("prompt", pi.get("question", ""))
                pt = pi.get("ground_truth", pi.get("answer", ""))
                if not pq or not pt:
                    continue
                pp = generate(edited_model, tok, pq, device=device)
                if contains_match(pp, pt):
                    p_hits += 1
                p_tot += 1
        if p_tot > 0:
            port_scores.append(100.0 * p_hits / p_tot)

        # ── Locality ──────────────────────────────────────────────────────
        # ROME edits weights — locality tests whether unrelated weights changed.
        # We generate from edited model and compare to ground_truth directly.
        l_hits, l_tot = 0, 0
        for _, loc_items in item.get("locality", {}).items():
            if not isinstance(loc_items, list):
                continue
            for li in loc_items:
                lq = li.get("prompt", li.get("question", ""))
                lt = li.get("ground_truth", li.get("answer", ""))
                if not lq or not lt:
                    continue
                lp = generate(edited_model, tok, lq, device=device)
                if contains_match(lp, lt):
                    l_hits += 1
                l_tot += 1
        if l_tot > 0:
            loc_scores.append(100.0 * l_hits / l_tot)

        # ── Restore original weights ──────────────────────────────────────
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(device)

    return {
        "edit_success": round(100.0 * edit_hits / total, 2),
        "portability":  round(float(np.mean(port_scores)), 2) if port_scores else None,
        "locality":     round(float(np.mean(loc_scores)),  2) if loc_scores else None,
    }


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(results, dataset_name):
    paper = PAPER_NUMBERS[dataset_name]
    print(f"\n{'='*62}")
    print(f"  ROME Baseline -- LLaMA-2-7B -- {dataset_name}")
    print(f"{'='*62}")
    print(f"  {'Metric':<20} {'Ours':>8}  {'Paper':>8}  {'Delta':>8}")
    print(f"  {'-'*48}")
    for key, label in [("edit_success", "Edit Success"),
                       ("portability",  "Portability"),
                       ("locality",     "Locality")]:
        ours   = results.get(key)
        theirs = paper.get(key)
        if ours is not None and theirs is not None:
            d = ours - theirs
            print(f"  {label:<20} {ours:>8.2f}  {theirs:>8.2f}  {('+' if d>=0 else '')}{d:>7.2f}")
        else:
            print(f"  {label:<20} {'N/A':>8}  {theirs:>8.2f}  {'N/A':>8}")
    print(f"{'='*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ZsRE", "WikiDatacounterfact"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--hparams",    default=HPARAMS)
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "results/tables"))
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data/rome_stats"), exist_ok=True)

    # Load hparams
    hparams = ROMEHyperParams.from_hparams(args.hparams)
    hparams.stats_dir = os.path.join(ROOT, "data/rome_stats")

    # Load dataset
    path = ZSRE_PATH if args.dataset == "ZsRE" else WIKICF_PATH
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[Data] {args.dataset}: {len(dataset)} samples (evaluating {args.max_samples})")

    # Load model
    model, tok = load_model(args.device)

    # Evaluate
    print("NOTE: First run computes Wikipedia statistics (~5-10 min, cached after).")
    results = evaluate(model, tok, hparams, dataset, args.max_samples, args.device)

    print_table(results, args.dataset)

    out = os.path.join(args.output_dir, f"rome_{args.dataset.lower()}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
