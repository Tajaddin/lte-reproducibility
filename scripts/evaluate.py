"""
LTE-LoRA Evaluation Script
============================
Evaluates a trained LTE-LoRA checkpoint on ZsRE and WikiDatacounterfact.
Uses the BM25 retrieval memory bank for the LTE inference phase.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/lte_lora_llama2 \
        --dataset ZsRE \
        --data_dir ./data \
        --output_dir ./results

    python scripts/evaluate.py \
        --checkpoint ./checkpoints/lte_lora_llama2 \
        --dataset WikiDatacounterfact \
        --data_dir ./data \
        --output_dir ./results
"""

import os
import sys
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_knowedit_dataset, prepare_edit_memory_bank
from src.metrics import (
    compute_edit_success, compute_portability,
    compute_locality, aggregate_results,
    print_results_table, compare_with_paper,
)
from src.retrieval import EditMemoryBank


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load QLoRA model + tokenizer from checkpoint."""
    base_model = "NousResearch/Llama-2-7b-chat-hf"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """Generate a single answer from the model."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_lte_prompt(edit: str, query: str) -> str:
    """Paper prompt format."""
    return (
        f"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
        f"[Updated Information]{edit} [Query]{query} [/INST]"
    )


def evaluate_dataset(
    model,
    tokenizer,
    dataset: list,
    memory_bank: EditMemoryBank,
    max_samples: int = None,
    device: str = "cuda",
) -> list:
    """Run LTE evaluation on a dataset split."""
    if max_samples:
        dataset = dataset[:max_samples]

    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        result = {}

        # ── Edit Success ──────────────────────────────────────────────────
        edit_desc = item.get("edit_descriptor", "")
        if not edit_desc:
            subj = item.get("subject", "")
            tgt  = item.get("target_new", "")
            edit_desc = f"{subj} is now {tgt}"

        prompt_q = item.get("prompt", "")
        target   = item.get("target_new", "")

        lte_prompt = build_lte_prompt(edit_desc, prompt_q)
        pred = generate_answer(model, tokenizer, lte_prompt, device=device)

        from src.metrics import contains_match, compute_portability
        result["edit_success"] = 100.0 if contains_match(pred, target) else 0.0

        # ── Portability ───────────────────────────────────────────────────
        port_queries = item.get("portability", {})
        port_preds, port_targets = [], []
        for port_type, port_items in port_queries.items():
            if isinstance(port_items, list):
                for pi in port_items:
                    pq = pi.get("prompt", pi.get("question", ""))
                    pt = pi.get("ground_truth", pi.get("answer", ""))
                    if pq and pt:
                        lp = build_lte_prompt(edit_desc, pq)
                        pp = generate_answer(model, tokenizer, lp, device=device)
                        port_preds.append(pp)
                        port_targets.append(pt)

        result["portability"] = (
            compute_portability(port_preds, port_targets)
            if port_preds else None
        )

        # ── Locality ──────────────────────────────────────────────────────
        # Locality: even with the current edit in context, the model should
        # still correctly answer unrelated questions (out-of-scope training).
        # Compare model output to ground_truth of locality items.
        loc_queries = item.get("locality", {})
        loc_preds, loc_targets = [], []
        for loc_type, loc_items in loc_queries.items():
            if isinstance(loc_items, list):
                for li in loc_items:
                    lq = li.get("prompt", li.get("question", ""))
                    lt = li.get("ground_truth", li.get("answer", ""))
                    if lq and lt:
                        # Ask locality question WITH edit in context —
                        # LTE's out-of-scope training means the model should
                        # ignore the irrelevant edit and answer from pre-training.
                        loc_prompt = build_lte_prompt(edit_desc, lq)
                        loc_pred = generate_answer(model, tokenizer, loc_prompt, device=device)
                        loc_preds.append(loc_pred)
                        loc_targets.append(lt)

        result["locality"] = (
            compute_portability(loc_preds, loc_targets)
            if loc_preds else None
        )

        # Fluency: placeholder (PPL-based, computed separately)
        result["fluency"] = None

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to LTE-LoRA checkpoint directory")
    parser.add_argument("--dataset", required=True,
                        choices=["ZsRE", "WikiDatacounterfact"])
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing KnowEdit dataset files")
    parser.add_argument("--output_dir", default="./results/tables")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation samples (for quick testing)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.device)

    # Load dataset
    dataset = load_knowedit_dataset(args.dataset, args.data_dir)

    # Build memory bank for retrieval
    edit_descs = prepare_edit_memory_bank(dataset)
    memory_bank = EditMemoryBank(edit_descs)

    # Evaluate
    raw_results = evaluate_dataset(
        model, tokenizer, dataset, memory_bank,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Aggregate
    summary = aggregate_results(raw_results)

    # Print
    print_results_table(summary, "LTE-LoRA (LLaMA2-Chat-7B)", args.dataset)
    compare_with_paper(summary, args.dataset)

    # Save
    out_path = os.path.join(
        args.output_dir,
        f"lte_lora_{args.dataset.lower()}_results.json"
    )
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": raw_results}, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
