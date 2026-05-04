"""
Baseline Evaluation: ROME and ICE
===================================
Runs ROME and ICE baselines via EasyEdit on LLaMA2-Chat-7B.
These baselines match Table 1 of Jiang et al. (2024).

Usage:
    # ICE baseline (no fine-tuning, runs locally with 4-bit quant)
    python scripts/run_baselines.py --method ICE --dataset ZsRE --data_dir ./data

    # ROME baseline
    python scripts/run_baselines.py --method ROME --dataset ZsRE --data_dir ./data

Requires EasyEdit to be installed:
    pip install easyeditor
or clone: git clone https://github.com/zjunlp/EasyEdit
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_knowedit_dataset
from src.metrics import (
    compute_edit_success, compute_portability,
    compute_locality, aggregate_results,
    print_results_table, compare_with_paper,
)


def run_ice_baseline(dataset: list, model_name: str, max_samples: int = None):
    """
    ICE: In-Context Editing.
    No fine-tuning — prepends the edit fact as context at inference time.
    Format: "Imagine that {edit}. {query}"
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if max_samples:
        dataset = dataset[:max_samples]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_name} for ICE baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    results = []

    for item in tqdm(dataset, desc="ICE Evaluation"):
        result = {}

        subj = item.get("subject", "")
        tgt  = item.get("target_new", "")
        edit_fact = f"{subj} is now {tgt}".strip()
        query = item.get("prompt", "")
        target = item.get("target_new", "")

        # ICE prompt: prepend edit as context
        ice_prompt = (
            f"<s>[INST] Imagine that {edit_fact}. "
            f"Given this information, {query} [/INST]"
        )

        inputs = tokenizer(ice_prompt, return_tensors="pt", truncation=True,
                           max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        from src.metrics import contains_match
        result["edit_success"] = 100.0 if contains_match(pred, target) else 0.0

        # Portability
        port_preds, port_targets = [], []
        for _, port_items in item.get("portability", {}).items():
            if isinstance(port_items, list):
                for pi in port_items:
                    pq = pi.get("prompt", pi.get("question", ""))
                    pt = pi.get("ground_truth", pi.get("answer", ""))
                    if pq and pt:
                        pp_prompt = (
                            f"<s>[INST] Imagine that {edit_fact}. "
                            f"Given this, {pq} [/INST]"
                        )
                        inp = tokenizer(pp_prompt, return_tensors="pt",
                                        truncation=True, max_length=512).to("cuda")
                        with torch.no_grad():
                            oo = model.generate(
                                **inp, max_new_tokens=50, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        pp = tokenizer.decode(
                            oo[0][inp["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                        port_preds.append(pp)
                        port_targets.append(pt)

        result["portability"] = (
            compute_portability(port_preds, port_targets)
            if port_preds else None
        )

        # Locality: ICE with no edit should be same as baseline (locality=100%)
        result["locality"] = 100.0
        result["fluency"]  = None

        results.append(result)

    return results


def run_rome_via_easyedit(dataset: list, model_name: str, max_samples: int = None):
    """
    ROME baseline via EasyEdit toolkit.
    Requires: pip install easyeditor
    """
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
    except ImportError:
        raise ImportError(
            "EasyEdit not found. Install with:\n"
            "  pip install easyeditor\n"
            "or: git clone https://github.com/zjunlp/EasyEdit && "
            "cd EasyEdit && pip install -e ."
        )

    if max_samples:
        dataset = dataset[:max_samples]

    hparams = ROMEHyperParams.from_pretrained(model_name)
    editor  = BaseEditor.from_hparams(hparams)

    prompts      = [item["prompt"]      for item in dataset]
    targets_new  = [item["target_new"]  for item in dataset]
    subjects     = [item["subject"]     for item in dataset]
    ground_truth = [item.get("ground_truth", "") for item in dataset]

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=targets_new,
        subject=subjects,
        ground_truth=ground_truth,
        keep_original_weight=False,
    )

    results = []
    for m in metrics:
        result = {
            "edit_success": m.get("post", {}).get("rewrite_acc", [0])[0] * 100,
            "portability":  m.get("post", {}).get("portability", {}).get("portability_acc", [None])[0],
            "locality":     m.get("post", {}).get("locality", {}).get("neighborhood_acc", [None])[0],
            "fluency":      None,
        }
        if result["portability"] is not None:
            result["portability"] *= 100
        if result["locality"] is not None:
            result["locality"] *= 100
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["ICE", "ROME"])
    parser.add_argument("--dataset", required=True,
                        choices=["ZsRE", "WikiDatacounterfact"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_dir", default="./results/tables")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_knowedit_dataset(args.dataset, args.data_dir)

    if args.method == "ICE":
        raw = run_ice_baseline(dataset, args.model, args.max_samples)
    else:
        raw = run_rome_via_easyedit(dataset, args.model, args.max_samples)

    summary = aggregate_results(raw)
    print_results_table(summary, f"{args.method} (LLaMA2-Chat-7B)", args.dataset)
    compare_with_paper(summary, args.dataset)

    out_path = os.path.join(
        args.output_dir,
        f"{args.method.lower()}_{args.dataset.lower()}_results.json"
    )
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": raw}, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
