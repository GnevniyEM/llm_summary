import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    "gazeta_train.jsonl",
    "gazeta_val.jsonl",
    "gazeta_test.jsonl",
]
#убираем ненужные колонки, дубликаты, выбросы
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["summary", "text"]].copy()

    df["summary"] = df["summary"].astype(str)
    df["text"] = df["text"].astype(str)

    df = df.drop_duplicates(subset=["text"], keep="first")
    df = df.drop_duplicates(subset=["summary"], keep="first")

    df = df[df["text"].str.len().between(2000, 8000)]
    df = df[df["summary"].str.len().between(150, 500)]

    return df.reset_index(drop=True)

for name in FILES:
    in_path = os.path.join(RAW_DIR, name)
    out_path = os.path.join(OUT_DIR, name)

    df = pd.read_json(in_path, lines=True)
    df = preprocess(df)

    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"{name}: {df.shape} -> saved to {out_path}")