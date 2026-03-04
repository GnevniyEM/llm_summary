# test_lora.py
import torch
import evaluate
from datasets import load_dataset
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm.auto import tqdm
import json
from datetime import datetime
from pathlib import Path
import click
import yaml  # нужно для чтения YAML


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", default="configs/test.yaml", show_default=True, help="Path to config")
def main(config: str):
    cfg = load_yaml(config)

    BASE_MODEL = cfg["BASE_MODEL"]
    LORA_DIR   = cfg["LORA_DIR"]
    DATA_FILE  = cfg["DATA_FILE"]

    N_TEST = cfg["N_TEST"]
    BATCH  = cfg["BATCH"]
    MAX_NEW = cfg["MAX_NEW"]
    max_seq_length = cfg["max_seq_length"]

    ds = load_dataset("json", data_files={"val": DATA_FILE})["val"].select(range(N_TEST))
    texts = ds["text"]
    refs  = ds["summary"]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    FastLanguageModel.for_inference(model)
    model.eval()

    def build_prompt(text: str) -> str:
        return f"Сделай краткий пересказ следующего текста:\n{text}\nПересказ:"

    @torch.inference_mode()
    def generate_summaries(model, tok, texts):
        prompts = [build_prompt(t) for t in texts]
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to("cuda")
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True)  # unsloth не поддерживает num_beams>1

        gen = out[:, inp["input_ids"].shape[1]:]
        preds = tok.batch_decode(gen, skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        return preds

    preds = []
    total_batches = (len(texts) + BATCH - 1) // BATCH
    for i in tqdm(range(0, len(texts), BATCH), total=total_batches, desc="Testing"):
        preds.extend(generate_summaries(model, tok, texts[i:i+BATCH]))

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs)
    print(scores)

    log_dir = Path("logs/test")
    log_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "lora_dir": LORA_DIR,
        "base_model": BASE_MODEL,
        "data_file": DATA_FILE,
        "N_TEST": N_TEST,
        "BATCH": BATCH,
        "MAX_NEW": MAX_NEW,
        "max_seq_length": max_seq_length,
        "scores": scores,
    }

    with (log_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()