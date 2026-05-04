"""
Data Utilities for LTE Reproducibility
========================================
Handles loading and formatting of:
  - ZsRE (zero-shot relation extraction)
  - WikiDatacounterfact
  - LTE alignment training data

Prompt format from paper:
    [Updated Information]{edit} [Query]{query}
"""

import json
import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset


LTE_PROMPT_TEMPLATE = "[Updated Information]{edit} [Query]{query}"

LLAMA2_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{prompt} [/INST] {response}</s>"""


def format_lte_prompt(edit: str, query: str) -> str:
    """Format a single (edit, query) pair using LTE prompt template."""
    return LTE_PROMPT_TEMPLATE.format(edit=edit, query=query)


def format_llama2_chat(prompt: str, response: str = "") -> str:
    """Wrap prompt in LLaMA2-Chat instruction format."""
    return LLAMA2_CHAT_TEMPLATE.format(prompt=prompt, response=response)


class LTEAlignmentDataset(Dataset):
    """
    Dataset for the LTE Alignment Phase (SFT).

    Each sample contains:
        - edit descriptor: the new fact to be learned
        - query: either in-scope or out-of-scope question
        - response: expected answer
        - scope: 'in_scope' | 'out_of_scope' | 'linguistic'

    Data format (from authors' GitHub):
    {
        "edit": "The capital of France is Berlin",
        "in_scope": [
            {"query": "What is the capital of France?", "answer": "Berlin"},
            ...
        ],
        "out_of_scope": [
            {"query": "What is the capital of Germany?", "answer": "Berlin"},
            ...
        ]
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r") as f:
            raw = json.load(f)

        for item in raw:
            edit = item["edit"]

            # In-scope samples
            for qa in item.get("in_scope", []):
                prompt = format_lte_prompt(edit, qa["query"])
                full = format_llama2_chat(prompt, qa["answer"])
                self.samples.append({"text": full, "scope": "in_scope"})

            # Out-of-scope samples
            for qa in item.get("out_of_scope", []):
                # Out-of-scope uses empty edit descriptor
                prompt = format_lte_prompt("", qa["query"])
                full = format_llama2_chat(prompt, qa["answer"])
                self.samples.append({"text": full, "scope": "out_of_scope"})

        print(f"[Data] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        # Labels = input_ids (causal LM, loss on all tokens)
        labels = input_ids.clone()
        # Mask padding tokens in loss
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_knowedit_dataset(dataset_name: str, data_dir: str) -> List[Dict]:
    """
    Load evaluation dataset in KnowEdit / EasyEdit format.

    Supports: ZsRE, WikiDatacounterfact
    Returns list of dicts with keys:
        prompt, target_new, ground_truth, portability, locality
    """
    path_map = {
        "ZsRE": os.path.join("knowedit", "benchmark", "ZsRE", "ZsRE-test-all.json"),
        "WikiDatacounterfact": os.path.join("knowedit", "benchmark", "wiki_counterfact", "test_cf.json"),
    }

    rel_path = path_map.get(dataset_name)
    if rel_path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(path_map.keys())}")

    fpath = os.path.join(data_dir, rel_path)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Dataset file not found: {fpath}\n"
            f"Download from: https://github.com/zjunlp/EasyEdit/tree/main/data"
        )

    with open(fpath, "r") as f:
        data = json.load(f)

    print(f"[Data] Loaded {len(data)} samples from {dataset_name}")
    return data


def prepare_edit_memory_bank(dataset: List[Dict]) -> List[str]:
    """
    Build the memory bank of edit descriptors for LTE inference phase.
    Each entry is the edit fact string used for BM25 retrieval.
    """
    bank = []
    for item in dataset:
        # Format: "subject relation target_new"
        edit_str = item.get("edit_descriptor", "")
        if not edit_str:
            subj = item.get("subject", "")
            rel  = item.get("relation", "")
            tgt  = item.get("target_new", "")
            edit_str = f"{subj} {rel} {tgt}".strip()
        bank.append(edit_str)
    return bank
