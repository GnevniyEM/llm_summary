from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os
from transformers.trainer_utils import get_last_checkpoint

import os
import click
import yaml 

os.makedirs("logs", exist_ok=True)
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", default="configs/train.yaml", show_default=True, help="Path to config")
def main(config: str):
    cfg = load_yaml(config)

    #CONFIG 
    max_seq_length = cfg["max_seq_length"]
    max_steps = cfg["max_steps"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    warmup_steps = cfg["warmup_steps"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    eval_step_size = cfg["eval_step_size"]
    seed = cfg["seed"]
    logging_steps = cfg["logging_steps"]
    save_steps = cfg["save_steps"]
    eval_steps = cfg["eval_steps"]
    weight_decay = cfg["weight_decay"]

    #Lora Paramets
    r = cfg["r"]
    lora_alpha = cfg["lora_alpha"]
    lora_dropout = cfg["lora_dropout"]
    base_model = cfg["base_model"]

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

            dataset_kwargs={"skip_prepare_dataset": True},
        )
    )

    last_ckpt = get_last_checkpoint("model")
    trainer.train(resume_from_checkpoint=last_ckpt) if last_ckpt else trainer.train()


if __name__ == "__main__":
    main()