# Emotional Support Dialogue Fine-Tuning

LLM fine-tuning pipeline for emotional support conversation systems. Supports two training stages: **SFT (Supervised Fine-Tuning)** and **DPO (Direct Preference Optimization)** alignment.

## Overview

This repository provides training code for building empathetic dialogue models using:
- **EmpatheticDialogues (ED)** — Facebook's empathetic dialogue dataset
- **ESConv (ESC)** — Emotional Support Conversation dataset

Supported base models: **LLaMA**, **Mistral**, **Qwen2**

## Repository Structure

```
.
├── preprocess_data_ED.py       # Preprocess EmpatheticDialogues dataset
├── preprocess_data_ESC.py      # Preprocess ESConv dataset
├── EVAD_sample.json            # Sample evaluation data
│
├── ACQ/                        # Stage 1: Supervised Fine-Tuning (SFT)
│   ├── train_full_llama.py
│   ├── train_full_mistral.py
│   ├── train_full_qwen.py
│   ├── ds_config_zero3.json    # DeepSpeed ZeRO-3 config
│   └── sft_exp_config/         # Per-model training configs
│       ├── ed_llama_onedialog.json
│       ├── ed_mistral_onedialog.json
│       ├── ed_qwen_onedialog.json
│       ├── esc_llama_onedialog.json
│       ├── esc_mistral_onedialog.json
│       └── esc_qwen_onedialog.json
│
├── ALI/                        # Stage 2: DPO Alignment
│   ├── dpo_peft_llama.py
│   ├── dpo_peft_mistral.py
│   ├── dpo_peft_qwen.py
│   ├── ds_config_zero3.json
│   └── dpo_exp_config/         # Per-model DPO configs
│       ├── llama_dpo.json
│       ├── mistral_dpo.json
│       └── qwen_dpo.json
│
└── chat_templates/             # Jinja chat templates for various models
```

## Setup

```bash
pip install torch transformers trl peft datasets easydict setproctitle wandb deepspeed
```

## Data Preprocessing

### EmpatheticDialogues

Place the raw dataset at `empatheticdialogues/train.csv`, then run:

```bash
python preprocess_data_ED.py
```

Output: `empatheticdialogues/prepro_train_w_idx.jsonl`

### ESConv

Place the raw dataset at `raw_data/test.txt`, then run:

```bash
python preprocess_data_ESC.py
```

Output: `ESC_dataset/prepro_test_w_idx_ver2.jsonl`

## Training

### Stage 1: SFT

```bash
cd ACQ
deepspeed train_full_qwen.py \
  --sft_exp_config esc_qwen_onedialog \
  --type ESC
```

Replace `qwen` with `llama` or `mistral` and adjust `--type` to `ED` or `ESC` as needed.

### Stage 2: DPO Alignment

```bash
cd ALI
deepspeed dpo_peft_qwen.py \
  --dpo_exp_config qwen_dpo \
  --type ESC
```

DPO uses LoRA (r=8, alpha=16) on `q_proj` and `v_proj` layers.

## Configuration

Edit `sft_exp_config/*.json` or `dpo_exp_config/*.json` to change:

| Parameter | Description |
|-----------|-------------|
| `model_path` | HuggingFace model ID or local path |
| `data_dir` | Directory containing processed datasets |
| `model_save_dir` | Output directory for saved checkpoints |
| `learning_rate` | Learning rate (default: 1e-6) |
| `num_train_epochs` | Number of training epochs |
| `max_seq_length` | Maximum sequence length (default: 1024) |

## Evaluation Data Format

`EVAD_sample.json` contains dialogue samples with `supporter` / `help-seeker` role pairs, used for preference evaluation in DPO training.

## WandB Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai). Set `run_name` in the config or disable with `report_to: none`.
