from framework.mitigation_framework import MitigationFramework
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch


if __name__ == '__main__':
    id2label = {0: "non-nurse", 1:"nurse"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mf = MitigationFramework()
    mf.initialize_mitigation_framework(id2label=id2label,
                                       use_case_name="nurse_vs_all")

    model_name_or_path = "saved_models/nurse_vs_all/bert-base-uncased/best_model/"

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    df = pd.read_csv("saved_datasets/test.csv")

    texts = df["cleaned_text"].tolist()[:300]

    label_ids_to_explain = [0, 1]

    output_dict = mf.run_explainer(model, tokenizer, texts, label_ids_to_explain, device=device)

    mf.run_identifier(output_dict)

    protected_attributes_dict = {0: ["he", "his", "him", "himself"],
                                 1: ["she", "her", "hers", "herself, nurse, nursing"]}


    df_train = pd.read_csv("saved_datasets/test.csv")
    df_train = df_train.iloc[:500]

    df_train_mitigated = mf.run_moderator(df_train, tokenizer, protected_attributes_dict,
                                          text_column_name="cleaned_text", label_column_name="label",
                                          mitigate_each_label_separately=False, batch_size=128)

    print(df_train_mitigated.head(30))
    df_train_mitigated.to_csv("saved_datasets/test_mitigated.csv", index=False)