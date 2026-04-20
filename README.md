# EVA: Empathetic LLMs with Emotional Validation

Official training code for **EVA**, a two-stage fine-tuning pipeline for empathetic dialogue generation grounded in **emotional validation theory**.

This repository contains the implementation of our **ACL Findings** paper:

> **Paper:** *I Don't Need Solution. I Need Emotional Support: Empathetic LLMs based on Emotional Validation*

---

## Overview

EVA improves emotional support response generation by aligning LLMs with emotional validation theory through two sequential stages:

| Stage | Name | Method | Description |
|-------|------|--------|-------------|
| 1 | **ACQ** (Empathy Acquisition) | SFT | Fine-tune on empathetic dialogue datasets to acquire general empathetic response ability |
| 2 | **ALI** (Emotional Validation Alignment) | DPO | Align model responses with emotional validation using the EVAD preference dataset |

**Supported base models:** LLaMA · Mistral · Qwen2

**Supported datasets:** EmpatheticDialogues (ED) · ESConv (ESC)

---

## Emotional Validation Levels

EVA is grounded in four operational levels of emotional validation. As the conversation progresses, the model moves from lower-level to deeper validation:

1. **Listening and Observing** — Acknowledging the speaker's presence and emotions
2. **Accurate Reflection** — Reflecting back what the speaker expressed
3. **Validating** — Affirming that the emotions make sense given the context
4. **Radical Genuineness** — Deep, authentic empathic engagement

---

## Repository Structure

```
EVA/
├── preprocess_data_ED.py       # Preprocess EmpatheticDialogues dataset
├── preprocess_data_ESC.py      # Preprocess ESConv dataset
├── EVAD_sample.json            # Sample preference data for ALI stage & evaluation
│
├── ACQ/                        # Stage 1: Empathy Acquisition (SFT)
│   ├── train_full_llama.py
│   ├── train_full_mistral.py
│   ├── train_full_qwen.py
│   ├── ds_config_zero3.json    # DeepSpeed ZeRO-3 config
│   └── sft_exp_config/         # Per-model SFT configs
│       ├── ed_llama_onedialog.json
│       ├── ed_mistral_onedialog.json
│       ├── ed_qwen_onedialog.json
│       ├── esc_llama_onedialog.json
│       ├── esc_mistral_onedialog.json
│       └── esc_qwen_onedialog.json
│
├── ALI/                        # Stage 2: Emotional Validation Alignment (DPO)
│   ├── dpo_peft_llama.py
│   ├── dpo_peft_mistral.py
│   ├── dpo_peft_qwen.py
│   ├── ds_config_zero3.json
│   └── dpo_exp_config/         # Per-model DPO configs
│       ├── llama_dpo.json
│       ├── mistral_dpo.json
│       └── qwen_dpo.json
│
└── chat_templates/             # Jinja chat templates for each base model
```

---

## Setup

Install the required packages:

```bash
pip install torch transformers trl peft datasets easydict setproctitle wandb deepspeed
```

---

## Data Preprocessing

### EmpatheticDialogues (ED)

Place the raw dataset file:
```
empatheticdialogues/train.csv
```

Run preprocessing:
```bash
python preprocess_data_ED.py
```

Output: `empatheticdialogues/prepro_train_w_idx.jsonl`

---

### ESConv (ESC)

Place the raw dataset file:
```
raw_data/test.txt
```

Run preprocessing:
```bash
python preprocess_data_ESC.py
```

Output: `ESC_dataset/prepro_test_w_idx_ver2.jsonl`

---

## Training

### Stage 1 — ACQ (Empathy Acquisition)

```bash
cd ACQ
deepspeed train_full_qwen.py \
  --sft_exp_config esc_qwen_onedialog \
  --type ESC
```

- Replace `qwen` with `llama` or `mistral` for other base models
- Set `--type` to `ED` or `ESC` depending on the dataset

---

### Stage 2 — ALI (Emotional Validation Alignment)

```bash
cd ALI
deepspeed dpo_peft_qwen.py \
  --dpo_exp_config qwen_dpo \
  --type ESC
```

DPO-based alignment using EVAD preference data. The current setup applies LoRA on:

| LoRA Config | Value |
|-------------|-------|
| Target modules | `q_proj`, `v_proj` |
| Rank (`r`) | 8 |
| Alpha | 16 |

---

## Configuration

Modify training settings via the JSON config files:

- `ACQ/sft_exp_config/*.json`
- `ALI/dpo_exp_config/*.json`

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `model_path` | Hugging Face model ID or local checkpoint path |
| `data_dir` | Directory containing preprocessed data |
| `model_save_dir` | Directory for saving checkpoints |
| `learning_rate` | Learning rate |
| `num_train_epochs` | Number of training epochs |
| `max_seq_length` | Maximum input sequence length |
| `run_name` | Weights & Biases run name |

---

## EVAD (Emotional Validation Aware Dataset)

`EVAD_sample.json` provides a sample of the EVAD format used in the ALI stage.

EVAD is a preference dataset where responses are ranked by how well they follow emotional validation theory across multi-turn conversations. Each sample contains:

- **Dialogue context** — the conversation history
- **Chosen response** — the preferred, more validating response
- **Rejected response** — the less validating response

---

## Evaluation

The paper introduces **EVAEval**, an evaluation metric that measures whether generated responses follow emotional validation across dialogue progression.

This repository currently includes:

- [x] Training pipelines for ACQ (Stage 1)
- [x] Preference alignment pipelines for ALI (Stage 2)
- [x] Sample EVAD-format data (`EVAD_sample.json`)

---

## Logging

Training metrics are logged with [Weights & Biases](https://wandb.ai).

- To enable: set `run_name` and related options in the config
- To disable: set `"report_to": "none"` in the config

---

## Citation

If you use this repository, please cite the EVA paper.
