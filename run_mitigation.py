from framework.mitigation_framework import MitigationFramework
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

if __name__ == '__main__':
    id2label = {0: "non-nurse", 1:"nurse"}

    mf = MitigationFramework()
    mf.initialize_mitigation_framework(id2label=id2label,
                                       use_case_name="nurse_vs_all")

    model_name_or_path = "saved_models/nurse_vs_all/bert-base-uncased/best_model/"

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    df = pd.read_csv("saved_datasets/test.csv")

    texts = df["cleaned_text"].tolist()[:300]

    label_ids_to_explain = [1]

    mf.run_explainer(model, tokenizer, texts, label_ids_to_explain)