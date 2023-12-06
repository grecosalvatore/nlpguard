from framework.mitigation_framework import MitigationFramework
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

if __name__ == '__main__':
    mf = MitigationFramework()
    mf.initialize_mitigation_framework(use_case_name="bert_toxicity")

    model_name_or_path = "saved_models/nurse_vs_all/bert-base-uncased/best_model/"

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    df = pd.read_csv("saved_datasets/test.csv")

    texts = df["cleaned_text"].tolist()[:300]

    mf.run_explainer(model, tokenizer, texts)