from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "phoenix120/gazeta-summaries"

def main(out_dir: str = "data/raw", dataset: str = DATASET):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(out), unzip=True, quiet=False)

    need = ["gazeta_train.jsonl", "gazeta_val.jsonl", "gazeta_test.jsonl"]
    missing = [f for f in need if not (out / f).is_file()]
    if missing:
        raise Exception("Не найдены файлы: " + ", ".join(missing))

    print(f"OK: данные в {out}")

if __name__ == "__main__":
    main()