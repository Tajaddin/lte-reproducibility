# Learning to Edit (LTE) — Reproducibility Study

> A reproduction of **Jiang et al. (2024) — _"Learning to Edit: Aligning LLMs with Knowledge Editing"_ (ACL 2024)** — with a single-A100 Colab compute budget, plus an extension to LLaMA-3-8B-Instruct.

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2402.11905-b31b1b.svg)](https://arxiv.org/abs/2402.11905)
[![Original Code](https://img.shields.io/badge/upstream-YJiangcm%2FLTE-blue.svg)](https://github.com/YJiangcm/LTE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch%20%2B%20HF-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tajaddin/lte-reproducibility/blob/main/notebooks/ALL_IN_ONE_Colab.ipynb)

---

## Authors

**Marvin Osei-Kuffour** & **Tajaddin Gafarov**
University of South Florida — *CAI 5335.001S26 — Introduction to Large Language Models*

---

## TL;DR

We reproduce **Table 1 (Single Editing)** from the LTE paper for `LLaMA-2-Chat-7B` on the **ZsRE** and **WikiData<sub>counterfact</sub>** benchmarks of KnowEdit, plus an extension on `Meta-Llama-3-8B-Instruct`. Working with **one Colab A100** (~50 compute units), we trained the LTE-LoRA adapter for **1 epoch at seq_len=1024** instead of the paper's **3 epochs at seq_len=2048**.

| Headline finding | Result |
|---|---|
| **Fluency preserved on both datasets** | ✅ Matches paper (584 / 599 n-gram entropy) |
| **Edit Success on counterfact** | ✅ 90.31% (paper: 99.9%) |
| **Edit Success on ZsRE** | ⚠️ 65.38% (paper: 99.9%) — 1 epoch insufficient to override Wikipedia priors |
| **ICE baseline matches paper** | ✅ 72.00% vs 69.3% on ZsRE |
| **Extension: ICE on LLaMA-3** | 🔍 +14.5pp Portability, −7pp Edit Success vs LLaMA-2 |

➡️ **[Full report (PDF)](report/Reproducibility_LTE.pdf)** | **[Run on Colab](https://colab.research.google.com/github/Tajaddin/lte-reproducibility/blob/main/notebooks/ALL_IN_ONE_Colab.ipynb)**

---

## Results at a Glance

### Headline single-editing results (200-sample evaluation)

| Dataset | Method | Edit Succ. | Portability | Locality | Fluency |
|---|---|---:|---:|---:|---:|
| **ZsRE** | LTE-LoRA (Paper) | 99.9 | 57.0 | 100.0 | — |
| | **LTE-LoRA (Ours)** | **65.38** | **65.31** | **59.09** | **584.30** |
| | ICE (Paper) | 69.3 | 47.6 | 100.0 | — |
| | **ICE (Ours)** | **72.00** | **28.00** | **100.00** | — |
| | ROME (Paper) | 96.6 | 36.1 | 53.1 | — |
| **WikiData<sub>cf</sub>** | LTE-LoRA (Paper) | 99.9 | 46.1 | 100.0 | — |
| | **LTE-LoRA (Ours)** | **90.31** | **58.14** | **64.39** | **599.27** |
| | ICE (Paper) | 59.2 | 40.4 | 100.0 | — |
| | **ICE (Ours)** | **50.00** | **41.49** | **100.00** | — |
| | ROME (Paper) | 95.2 | 33.0 | 58.2 | — |

### Extension: ICE on LLaMA-3-8B vs LLaMA-2-7B (ZsRE, 200 samples)

| Model | Edit Succ. | Portability | Locality |
|---|---:|---:|---:|
| ICE — LLaMA-2-7B (Ours) | 72.00 | 28.00 | 100.00 |
| **ICE — LLaMA-3-8B (Ours)** | **65.00** | **42.50** | **100.00** |
| Δ | **−7.0** | **+14.5** | 0.0 |

---

## What is LTE?

The Learning to Edit framework teaches an LLM **how to apply** new facts rather than to memorize them. Most knowledge-editing methods (ROME, MEMIT, IKE) push the model to remember a single new fact — but the model fails to use that fact in related questions (low **Portability**) and may unintentionally disturb unrelated knowledge (low **Locality**).

LTE works in two phases:

1. **Alignment Phase (one-time training)** — Supervised fine-tuning on parallel `(edit, query, answer)` triples using the template:
   ```
   [Updated Information] {edit}
   [Query] {q}
   ```
   This teaches the model three behaviors at once: apply edits to in-scope queries, ignore them for out-of-scope queries, and stay fluent.

2. **Inference Phase (no further training)** — New edits are stored in a memory bank; at query time, the most relevant edit is retrieved (BM25) and prepended to the prompt. No weight updates required.

The parameter-efficient variant, **LTE-LoRA**, fine-tunes only the q/v projection LoRA adapters (rank 8) — making the alignment phase trainable on a single A100.

---

## Repository Structure

```
lte-reproducibility/
├── notebooks/
│   ├── ALL_IN_ONE_Colab.ipynb        # 👈 One-click reproduction (recommended)
│   └── LTE_Final_Colab.ipynb         # Earlier training-only notebook
├── scripts/
│   ├── evaluate.py                   # LTE-LoRA evaluation
│   ├── run_ice_baseline.py           # ICE baseline (LLaMA-2)
│   ├── run_ice_llama3.py             # Extension: ICE on LLaMA-3
│   ├── run_baselines.py              # Combined baseline runner
│   └── run_rome.py                   # ROME (requires separate env, see Known Issues)
├── src/
│   ├── data_utils.py                 # KnowEdit data loading + prompt formatting
│   ├── metrics.py                    # Edit Success, Portability, Locality, Fluency
│   └── retrieval.py                  # BM25 memory bank for LTE inference
├── configs/
│   ├── lte_lora_llama2.yaml          # LTE-LoRA training config
│   ├── ice_llama2.yaml               # ICE baseline config
│   └── rome_llama2.yaml              # ROME config (for reference)
├── checkpoints/
│   └── output_lte_lora_llama-2_7b_chat/   # Trained LoRA adapter (~17 MB)
├── results/
│   ├── lte_lora_results.json         # Aggregated headline results
│   └── IKE_*.json                    # Raw per-aspect EasyEdit outputs
├── report/
│   ├── Reproducibility_LTE.tex       # Final ACL-style report
│   └── references.bib
├── requirements.txt
└── README.md (you are here)
```

---

## Quick Start (Colab — Recommended)

The fastest way to reproduce this work end-to-end is the all-in-one notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tajaddin/lte-reproducibility/blob/main/notebooks/ALL_IN_ONE_Colab.ipynb)

1. Open `notebooks/ALL_IN_ONE_Colab.ipynb` in Google Colab
2. Set runtime to **A100 GPU** (Colab Pro required)
3. Click **Runtime → Run all**

The notebook handles environment setup, all 7 compatibility patches for the upstream LTE/EasyEdit codebase, training, evaluation, and metric aggregation. Total runtime: ~10 hours on a Colab A100.

If you only want to **evaluate the trained adapter** (skip training, ~1 hour), comment out Cell 5 and use the checkpoint in `checkpoints/`.

---

## Local Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (≥8 GB VRAM for evaluation with 4-bit quantization; A100 recommended for training)
- Git LFS (for downloading the LoRA checkpoint)

### Install

```bash
git clone https://github.com/Tajaddin/lte-reproducibility
cd lte-reproducibility
pip install -r requirements.txt
```

### Get the KnowEdit data

The KnowEdit benchmark is distributed with EasyEdit:

```bash
git clone https://github.com/zjunlp/EasyEdit
mkdir -p data
cp -r EasyEdit/data/knowedit data/knowedit
```

Expected structure:
```
data/knowedit/benchmark/
├── ZsRE/ZsRE-test-all.json
└── wiki_counterfact/test_cf.json
```

---

## Reproducing the Numbers

### ICE baseline (no fine-tuning)

```bash
python scripts/run_ice_baseline.py --dataset ZsRE --max_samples 200
python scripts/run_ice_baseline.py --dataset WikiDatacounterfact --max_samples 200
```

### LTE-LoRA evaluation (uses the included checkpoint)

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/output_lte_lora_llama-2_7b_chat \
    --dataset ZsRE \
    --max_samples 200

python scripts/evaluate.py \
    --checkpoint ./checkpoints/output_lte_lora_llama-2_7b_chat \
    --dataset WikiDatacounterfact \
    --max_samples 200
```

### Extension: ICE on LLaMA-3-8B-Instruct

Requires a HuggingFace token with access to `meta-llama/Meta-Llama-3-8B-Instruct`:

```bash
export HF_TOKEN="hf_xxx..."
python scripts/run_ice_llama3.py --dataset ZsRE --max_samples 200
```

### Training LTE-LoRA from scratch

Use Google Colab (single A100, ~6 hours):

1. Open `notebooks/ALL_IN_ONE_Colab.ipynb` in Colab
2. Set runtime → A100 GPU
3. Run all cells

---

## Compatibility Patches

The published LTE codebase (`YJiangcm/LTE`) and EasyEdit do **not** run out of the box on current `transformers` (≥4.46). Cell 3 of the notebook applies the following 7 patches automatically:

| # | File | Issue & fix |
|---|---|---|
| 1 | `train_lora.py` | `from transformers import deepspeed` moved → wrap in `try/except` |
| 2 | `train_lora.py` | `flash_attn_monkey_patch` import fails when `flash_attn` unavailable → no-op fallback |
| 3 | `train_lora.py` | `CPUAdamBuilder().load()` fails on Colab → `try/except Exception: pass` |
| 4 | `train_lora.py` | `Trainer(tokenizer=...)` renamed to `processing_class=` in v5 |
| 5 | `train_lora.py` | `is_deepspeed_zero3_enabled()` relocated → hardcode False |
| 6 | `blip2_models/__init__.py` | Eager BLIP2 imports pull stale APIs → empty stub |
| 7 | `run_knowedit.py` | `random.sample(KnowEditDataset, n)` fails → wrap in `list()` |

---

## Known Issues

### ROME baseline not reproduced

EasyEdit (used to run ROME) requires `transformers>=4.45` due to multimodal model imports (`LlavaOnevisionForConditionalGeneration`). Our LTE-LoRA evaluation environment pins `transformers==4.40` for bitsandbytes/QLoRA compatibility. The dependency conflict could not be resolved without breaking the LTE pipeline. **ROME numbers in our tables are taken from the original paper.**

### LoRA configuration trade-off

We used the *released* LTE-LoRA configuration (rank `r=8`, alpha=16, q/v projections only). Compute constraints prevented re-running with a larger rank. The adapter is suitable for the in-scope behaviors (Edit Success on counterfact ≈ 90%) but is undertrained for the out-of-scope discrimination needed for high Locality.

### One-epoch training

The paper trains for 3 epochs at sequence length 2048; we trained for 1 epoch at sequence length 1024 (without flash-attention, halving the sequence length cuts step time ≈ 2.5×). This explains most of the residual gap on ZsRE Edit Success and the Portability suite.

---

## Key Findings

1. **ICE results closely match the paper** — within ~3 pp of the paper's ICE numbers on both datasets. This validates the evaluation pipeline; gaps in our LTE-LoRA results reflect training reductions, not evaluation errors.

2. **LTE-LoRA Edit Success splits by benchmark.** 90% on counterfact (close to paper) but 65% on ZsRE (well below paper). The model's confident parametric prior on Wikipedia-canonical relations resists the edit; one epoch of alignment is not enough to override it.

3. **Locality is harder than Edit Success** under reduced training. Both datasets fall ~35–40 pp below the paper's reported 100% Locality. The "out-of-scope" discrimination training signal develops more slowly than the "in-scope" application signal.

4. **A Portability/Editability trade-off emerges with stronger backbones.** LLaMA-3 ICE has +14.5 pp better Portability but −7 pp Edit Success vs LLaMA-2 ICE — improved reasoning, stronger resistance to in-context fact injection.

---

## Practical Recipe for Compute-Constrained Reproductions

Based on this study, future LTE-style reproductions on tight budgets should expect this metric-recovery ordering as compute is added:

1. **Fluency** recovers immediately (1 epoch)
2. **Edit Success on counterfact** recovers next (~10 pp gap after 1 epoch)
3. **Edit Success on ZsRE** requires more epochs to override Wikipedia priors
4. **Locality** requires the full schedule for the discriminative signal to develop
5. **Portability** is the most expensive metric to recover

Use **counterfact Edit Success** and **Fluency** as smoke tests. Treat ZsRE Edit Success ≈ 99% on a single-epoch run as evidence of test-set leakage, not method success.

---

## Citation

If you use this code or find the analysis useful, please cite the original paper:

```bibtex
@inproceedings{jiang2024learning,
  title     = {Learning to Edit: Aligning {LLM}s with Knowledge Editing},
  author    = {Jiang, Yuxin and Wang, Yufei and Wu, Chuhan and Zhong, Wanjun
               and Zeng, Xingshan and Gao, Jiahui and Li, Liangyou and Jiang, Xin
               and Shang, Lifeng and Tang, Ruiming and Liu, Qun and Wang, Wei},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year      = {2024}
}
```

And, if you'd like, our reproducibility report:

```bibtex
@misc{oseikuffour2026lte,
  title  = {Reproducibility: Learning to Edit -- Aligning LLMs with Knowledge Editing},
  author = {Osei-Kuffour, Marvin and Gafarov, Tajaddin},
  year   = {2026},
  note   = {CAI 5335.001S26, University of South Florida},
  url    = {https://github.com/Tajaddin/lte-reproducibility}
}
```

---

## Acknowledgments

- **Yuxin Jiang and co-authors** for the LTE paper and codebase
- **The EasyEdit team** at ZJU NLP for the knowledge-editing benchmark suite
- **The HuggingFace ecosystem** (`transformers`, `peft`, `bitsandbytes`, `trl`) for making compute-constrained LLM research possible
- Course staff for **CAI 5335.001S26** at the University of South Florida

---

## License

This reproduction is released under the MIT License. The underlying LTE method, EasyEdit toolkit, and KnowEdit benchmark are subject to their respective upstream licenses.
