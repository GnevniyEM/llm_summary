# test_lora.py
import torch
import evaluate
from datasets import load_dataset
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm.auto import tqdm

BASE_MODEL = "unsloth/Qwen2.5-0.5B-bnb-4bit"
LORA_DIR   = "model/checkpoint-3000"             
DATA_FILE  = "data/preprocessed/gazeta_test.jsonl"  

N_TEST = 1000
BATCH  = 8
MAX_NEW = 170
max_seq_length=2800

def build_prompt(text: str) -> str:
    return f"Сделай краткий пересказ следующего текста:\n{text}\nПересказ:"

@torch.inference_mode()
def generate_summaries(model, tok, texts):
    prompts = [build_prompt(t) for t in texts]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inp = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to("cuda")
    out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1 , use_cache=True) #unsloth не поддерживает num_beams>1

    gen = out[:, inp["input_ids"].shape[1]:]
    preds = tok.batch_decode(gen, skip_special_tokens=True)
    preds = [p.strip() for p in preds]
    return preds

def main():
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

    preds = []
    total_batches = (len(texts) + BATCH - 1) // BATCH
    for i in tqdm(range(0, len(texts), BATCH), total=total_batches, desc="Testing"):
        preds.extend(generate_summaries(model, tok, texts[i:i+BATCH]))

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=preds, references=refs)
    print(scores)

if __name__ == "__main__":
    main()