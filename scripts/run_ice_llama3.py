"""
Extension: ICE Baseline with Meta-LLaMA-3-8B-Instruct
=======================================================
Runs ICE (In-Context Editing) on LLaMA-3-8B-Instruct to compare
cross-generation ICE performance vs LLaMA-2-7b-chat-hf.

No fine-tuning needed. Uses 4-bit quantization.
Requires HF token with LLaMA-3 access.

Usage:
    python scripts/run_ice_llama3.py --dataset ZsRE
    python scripts/run_ice_llama3.py --dataset WikiDatacounterfact
"""

import os, sys, json, argparse, re, string
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ZSRE_PATH   = "./data/knowedit/benchmark/ZsRE/ZsRE-test-all.json"
WIKICF_PATH = "./data/knowedit/benchmark/wiki_counterfact/test_cf.json"
MODEL_NAME  = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")  # set via `export HF_TOKEN=...` before running

# LLaMA-2 ICE results for comparison
LLAMA2_RESULTS = {
    "ZsRE":               {"edit_success": 72.00, "portability": 28.00, "locality": 100.00},
    "WikiDatacounterfact": {"edit_success": 50.00, "portability": 41.49, "locality": 100.00},
}
PAPER_ICE = {
    "ZsRE":               {"edit_success": 69.3,  "portability": 47.6,  "locality": 100.0},
    "WikiDatacounterfact": {"edit_success": 59.2,  "portability": 40.4,  "locality": 100.0},
}


# ── Helpers ─────────────────────────────────────────────────────────────────

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

def ice_prompt_llama3(edit_fact: str, query: str) -> str:
    """LLaMA-3 Instruct chat format with ICE prefix."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a helpful assistant. Answer concisely.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Imagine that {edit_fact}. Given this updated information, answer concisely.\n\n"
        f"{query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=40, device="cuda"):
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
        add_special_tokens=False,
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


# ── Load model ───────────────────────────────────────────────────────────────

def load_model(device="cuda"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_NAME} (4-bit QLoRA)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    model.eval()
    print(f"Model ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tok


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model, tok, dataset, max_samples, device="cuda"):
    if max_samples:
        dataset = dataset[:max_samples]

    edit_hits, total = 0, len(dataset)
    port_scores, loc_scores = [], []

    for item in tqdm(dataset, desc="ICE-LLaMA3 Evaluation"):
        subj  = item.get("subject", "")
        tgt   = item.get("target_new", "")
        query = item.get("prompt", "")
        edit  = f"{subj} is now {tgt}"

        # Edit Success
        pred = generate(model, tok, ice_prompt_llama3(edit, query), device=device)
        if contains_match(pred, tgt):
            edit_hits += 1

        # Portability
        p_hit, p_tot = 0, 0
        for _, items in item.get("portability", {}).items():
            if not isinstance(items, list):
                continue
            for pi in items:
                pq = pi.get("prompt", pi.get("question", ""))
                pt = pi.get("ground_truth", pi.get("answer", ""))
                if not pq or not pt:
                    continue
                pp = generate(model, tok, ice_prompt_llama3(edit, pq), device=device)
                if contains_match(pp, pt):
                    p_hit += 1
                p_tot += 1
        if p_tot > 0:
            port_scores.append(100.0 * p_hit / p_tot)

        # Locality: 100% by design (no weight changes)
        loc_scores.append(100.0)

    import numpy as np
    return {
        "edit_success": round(100.0 * edit_hits / total, 2),
        "portability":  round(float(np.mean(port_scores)), 2) if port_scores else None,
        "locality":     100.0,
    }


# ── Print table ──────────────────────────────────────────────────────────────

def print_table(results, dataset_name):
    llama2 = LLAMA2_RESULTS[dataset_name]
    paper  = PAPER_ICE[dataset_name]
    print(f"\n{'='*72}")
    print(f"  Extension: ICE Baseline — LLaMA-3-8B-Instruct vs LLaMA-2-7b-chat")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*72}")
    print(f"  {'Metric':<20} {'LLaMA-3':>10}  {'LLaMA-2':>10}  {'Paper ICE':>10}  {'D vs L2':>8}")
    print(f"  {'-'*60}")
    for key, label in [("edit_success","Edit Success"), ("portability","Portability"), ("locality","Locality")]:
        ours   = results.get(key)
        l2     = llama2.get(key)
        pap    = paper.get(key)
        if ours is not None and l2 is not None:
            delta = ours - l2
            print(f"  {label:<20} {ours:>10.2f}  {l2:>10.2f}  {pap:>10.2f}  {('+' if delta>=0 else '')}{delta:>7.2f}")
        else:
            print(f"  {label:<20} {'N/A':>10}  {l2:>10.2f}  {pap:>10.2f}  {'N/A':>8}")
    print(f"{'='*72}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ZsRE", "WikiDatacounterfact"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="./results/tables")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    path = ZSRE_PATH if args.dataset == "ZsRE" else WIKICF_PATH
    with open(path) as f:
        dataset = json.load(f)
    print(f"[Data] {args.dataset}: {len(dataset)} samples (evaluating {args.max_samples})")

    model, tok = load_model(args.device)
    results = evaluate(model, tok, dataset, args.max_samples, args.device)

    print_table(results, args.dataset)

    out = os.path.join(args.output_dir, f"ice_llama3_{args.dataset.lower()}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
