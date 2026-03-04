
# Unsloth + QLoRA Fine-Tuning Project

This project is for practical training in fine-tuning an unsloth+qlora model. The base model used for fine-tuning is `unsloth/Qwen2.5-0.5B-bnb-4bit`.

## Dataset
The dataset for training is the [Gazeta Summaries dataset](https://www.kaggle.com/datasets/phoenix120/gazeta-summaries).

## Installation

1. Install the required dependencies by running the following command:
    
    pip install -r requirements.txt
    

2. To download the dataset, run the script `download_data.py`:
    
    python scripts/download_data.py
    

3. For baseline evaluation (ROUGE metrics), run the `baseline.py` script:
    
    python scripts/baseline.py
    

4. Preprocess the dataset using:
    
    python scripts/preprocess.py
    

5. To train the model, run the `train_script.py`:
    
    python scripts/train_script.py
    

6. After training, run the `test_script.py` to evaluate the model:
    
    python scripts/test_script.py
    

## Project Structure

```
data/
├── preprocessed/
│── raw/
logs/
model/
notebooks/
├── eda.ipynb
scripts/
├── __pycache__/
├── baseline.py
├── download_data.py
├── preprocess.py
├── test_script.py
└── train_script.py
```

## Final Metrics

### Baseline:
- ROUGE-1: 0.1779
- ROUGE-2: 0.0610
- ROUGE-L: 0.1730

### Fine-Tuned Model:
- ROUGE-1: 0.2112
- ROUGE-2: 0.0673
- ROUGE-L: 0.2056

