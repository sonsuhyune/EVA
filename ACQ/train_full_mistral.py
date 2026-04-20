import os
import torch
import json
import wandb
import argparse
import torch.nn.functional as F
from easydict import EasyDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from datasets import Dataset
from setproctitle import setproctitle
from peft import LoraConfig

os.environ["WANDB_DIR"] = os.path.abspath("./")
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['TRANSFORMERS_CACHE'] = 'hf_cache'

# from huggingface_hub import login
# login()
def main(args, parsed_args):
    setproctitle(args.setproctitle)

    # Generate dataset based on the type of processing required
    def gen_dataset(raw_data, max_len):  # Modify the input to chat template format.
        if parsed_args.type == "ED":
            turns = {'input_ids': []}
            for dialog in raw_data:
                if dialog['inputs'][0]["role"] == "assistant":
                    dialog['inputs'] = dialog['inputs'][1:]
                input_dia = []
                input_dia.append({"role": "system",
                             "content": "You are a helpful and considerate supporter. Your job is to generate appropriate emotional supporting utterances based on the help-seeker's utterance. Please generate the supporter’s utterance based on the given dialogue history."})
                input_dia.extend(dialog['inputs'])
                chat_template_turn = tokenizer.apply_chat_template(input_dia, add_generation_prompt=True,
                                                                       max_length=args.max_seq_length, truncation=True)
                turns['input_ids'].append(chat_template_turn)
            dataset = Dataset.from_dict(turns)
            return dataset
        elif  parsed_args.type == "ESC":
            turns = {'input_ids': []}
            for dialog in raw_data:
                for turn in dialog['inputs']:
                    if turn[0]["role"] == "assistant":
                        turn = turn[1:]
                    input_dia = []
                    input_dia.append({"role": "system",
                                      "content": "You are a helpful and considerate supporter. Your job is to generate appropriate emotional supporting utterances based on the help-seeker's utterance. Please generate the supporter’s utterance based on the given dialogue history."})
                    input_dia.extend(turn)
                    chat_template_turn = tokenizer.apply_chat_template(input_dia, add_generation_prompt=True,
                                                                   truncation=True, max_length=max_len)

                    turns['input_ids'].append(chat_template_turn)
            dataset = Dataset.from_dict(turns)
            return dataset

    # Load tokenizer and set special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, add_eos_token=True,cache_dir="/data/ssh5131/hf_cache")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id == tokenizer.pad_token_id:
        if tokenizer.unk_token_id is None:
            raise NotImplementedError("Unexpected situation. Assign pad_token_id to something. (but not eos_token_id)")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print(tokenizer.pad_token_id,tokenizer.unk_token_id, tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    chat_template = open('/chat_templates/mistral-instruct.jinja').read() # Load chat template for system prompt utilization
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    tokenizer.chat_template = chat_template
    # Define a data collator for the model
    collator = DataCollatorForCompletionOnlyLM(instruction_template=args.instruction_template,
                                               response_template=args.response_template,
                                               tokenizer=tokenizer,
                                               mlm=False, )
    # Load training and validation data
    with open(os.path.join(args.data_dir, "{}.json".format(args.train_data)), "r") as f:
        train_raw_data = json.load(f)

    with open(os.path.join(args.data_dir, "{}.json".format(args.valid_data)), "r") as f:
        valid_raw_data = json.load(f)

    # Generate datasets for training and validation
    train_dataset = gen_dataset(train_raw_data,args.max_seq_length)
    print(train_dataset)
    valid_dataset = gen_dataset(valid_raw_data,args.max_seq_length)

    # Configure training arguments
    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        max_seq_length=args.max_seq_length,
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
        dataset_text_field='text',
        seed=args.seed,
        bf16=True,
        deepspeed=args.deepspeed,
        run_name=args.run_name,
        dataloader_num_workers=args.workers
    )
    # Load pre-trained model for causal language modeling
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                                 # cache_dir=args.cache_dir
                                                 )
    # Initialize the trainer with model, datasets, and training arguments
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )
    # Start training and save the fine-tuned model and tokenizer
    trainer.train()
    trainer.model.save_pretrained(os.path.join(args.model_save_dir, args.sft_model_name))
    tokenizer.save_pretrained(os.path.join(args.model_save_dir, args.sft_model_name))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sft_exp_config", dest="config_file", type=str,
                            help="Config json file")
    arg_parser.add_argument("--type", type=str,
                            help="Config json file")
    parsed_args = arg_parser.parse_args()
    # Load configuration from JSON file
    with open(os.path.join("./sft_exp_config", "{}.json").format(parsed_args.config_file)) as f:
        args = EasyDict(json.load(f))
    main(args,parsed_args)