from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os
from transformers.trainer_utils import get_last_checkpoint

import os
os.makedirs("logs", exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs"

#CONFIG
max_seq_length = 2800
max_steps = 3000
batch_size = 16
lr = 1e-5
warmup_steps = 100
gradient_accumulation_steps = 1
eval_step_size = 100
seed = 3407
logging_steps = 100
save_steps = 100
eval_steps = 100
weight_decay=0.01

#Lora Paramets
r = 16
lora_alpha = 32
lora_dropout = 0.05

base_model = "unsloth/Qwen2.5-0.5B-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

ds = load_dataset(
    "json",
    data_files={
        "train": "data/preprocessed/gazeta_train.jsonl",
        "validation": "data/preprocessed/gazeta_val.jsonl",
    },
)

def tokenize_pc(ex):
    prompt = f"Сделай краткий пересказ следующего текста:\n{ex['text']}\nПересказ:"
    summary = ex["summary"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    summary_ids = tokenizer(summary, add_special_tokens=False).input_ids
    summary_ids = summary_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + summary_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + summary_ids

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_ds = ds["train"].map(tokenize_pc, remove_columns=ds["train"].column_names)
eval_ds  = ds["validation"].map(tokenize_pc, remove_columns=ds["validation"].column_names)
eval_ds = eval_ds.shuffle(seed=seed).select(range(min(eval_step_size, len(eval_ds))))

model = FastLanguageModel.get_peft_model(
    model,
    r = r,
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    use_gradient_checkpointing = "unsloth",
    random_state = seed,
    max_seq_length = max_seq_length
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_steps = max_steps,
        max_seq_length = max_seq_length,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,

        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir="model",

        eval_strategy="steps",
        eval_steps=eval_steps,

        seed = seed,
        optim="adamw_8bit",
        learning_rate = lr,
        lr_scheduler_type = "cosine",
        warmup_steps = warmup_steps,
        weight_decay=weight_decay,

        report_to="tensorboard",
        logging_dir = "logs" ,

        # КЛЮЧЕВОЕ: чтобы Unsloth не требовал formatting_func
        dataset_kwargs={"skip_prepare_dataset": True},
    )
)

last_ckpt = get_last_checkpoint("model")
trainer.train(resume_from_checkpoint=last_ckpt) if last_ckpt else trainer.train()