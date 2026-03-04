#Для референса создаем baseline. 
#В качестве summary будем первое и последнее предложение текста и оценивать по Rouge. 
#В дальнейшем будем сравнивать результаты fine-tining c этими 

import pandas as pd
from rouge_score import rouge_scorer
import re 

df = pd.read_json('data/preprocessed/gazeta_test.jsonl', lines=True)

def get_first_and_last_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences[0] + " " + sentences[-1]  # Возвращаем первое и последнее предложение

df['baseline_summary'] = df['text'].apply(get_first_and_last_sentence)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
rouge_scores = []
for i in range(len(df)):
    reference = df['summary'][i]
    generated = df['baseline_summary'][i]
    scores = scorer.score(reference, generated)
    rouge_scores.append(scores)

rouge_1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
rouge_2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
rouge_L = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

print(f"ROUGE-1: {rouge_1:.4f}")
print(f"ROUGE-2: {rouge_2:.4f}")
print(f"ROUGE-L: {rouge_L:.4f}")