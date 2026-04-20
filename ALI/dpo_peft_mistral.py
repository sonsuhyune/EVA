import os
import torch
import json
import wandb
import argparse
import torch.nn.functional as F
from easydict import EasyDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer
from datasets import Dataset
from setproctitle import setproctitle
from peft import LoraConfig
os.environ["WANDB_DIR"] = os.path.abspath("./")
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# from huggingface_hub import login
# login()
# Main function to execute the training process

def main(args, parsed_args):
    setproctitle(args.setproctitle)

    def chat(dialog): ## Concatenate messages from the same role to form a continuous dialogue
        dia = []
        last = dialog[0]['role']
        dia.append(dialog[0])
        for i in dialog[1:]:

            if last == i['role']: # If the role is the same, concatenate the message content
                dia[-1] = {"content": dia[-1]["content"] + ". " + i["content"], "role": last}
            else:
                dia.append(i)
            last = dia[-1]["role"]
        if dia[0]['role'] == "assistant":
            dia = dia[1:]
        return dia
    def create_triplets(example):
        """Create the triplets (prompt, chosen, rejected)"""
        input_dia = [{"role": "system", "content": "You are a helpful and empathetic assistant."}]
        input_dia.extend(chat(example["history"]))
        input_dia.append({"role": "assistant", "content": ""})
        # Create a dictionary with prompt, chosen, and rejected responses
        tmp_dict = {
            "prompt": tokenizer.apply_chat_template(input_dia, max_length=args.max_seq_length, truncation=True,
                                                    tokenize=False),
            "chosen": example["Chosen"],
            "rejected": example["Rejected"]
        }





        return tmp_dict

    # Initialize the tokenizer with the specified model path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id == tokenizer.pad_token_id:
        if tokenizer.unk_token_id is None:
            raise NotImplementedError("Unexpected situation. Assign pad_token_id to something. (but not eos_token_id)")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print(tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.eos_token_id,
          tokenizer.convert_tokens_to_ids("</s>"))


    tokenizer.padding_side = 'left'  # to prevent errors with FA
    tokenizer.truncation_side = 'left'  # to prevent cutting off last generation

    # Load the training dataset from a JSON file
    train_dataset = load_dataset('json', data_files=os.path.join(args.data_dir, "{}.json".format(args.train_data)), split="train")
    train_original_columns = train_dataset.column_names
    print(train_dataset)

    train_dataset = train_dataset.map(
        create_triplets,
        remove_columns=train_original_columns
    )
    # Load the validation dataset from a JSON file
    valid_dataset = load_dataset('json', data_files=os.path.join(args.data_dir, "{}.json".format(args.valid_data)),
                                 split="train")
    valid_original_columns = valid_dataset.column_names
    print(valid_dataset)

    valid_dataset = valid_dataset.map(
        create_triplets,
        remove_columns=valid_original_columns
    )

    # Set training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        # max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        output_dir=os.path.join(args.model_save_dir, args.sft_model_name),
        report_to='wandb',
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # dataset_text_field='text',
        seed=args.seed,
        bf16=True,
        deepspeed=args.deepspeed,
        run_name=args.run_name,
        dataloader_num_workers=args.workers
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,

                                                 )
    model.config.use_cache = False
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    )
    # Configure the LoRA for fine-tuning
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Initialize the DPOTrainer for fine-tuning
    max_seq_length = 1024
    trainer = DPOTrainer(
        model,
        model_ref,
        train_dataset=train_dataset,

        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=max_seq_length,
        peft_config=peft_config,

    )
    # Start training and save the fine-tuned model and tokenizer
    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.model_save_dir, args.sft_model_name))
    tokenizer.save_pretrained(os.path.join(args.model_save_dir, args.sft_model_name))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dpo_exp_config", dest="config_file", type=str,
                            help="Config json file")
    arg_parser.add_argument("--type", type=str,
                            help="Config json file")
    parsed_args = arg_parser.parse_args()
    # Load configuration from JSON file
    with open(os.path.join("./dpo_exp_config", "{}.json").format(parsed_args.config_file)) as f:
        args = EasyDict(json.load(f))
    main(args,parsed_args)