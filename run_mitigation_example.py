from framework.mitigation_framework import MitigationFramework
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch


if __name__ == '__main__':

    # Mapping label ids to label names
    id2label = {0: "non-nurse", 1:"nurse"}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the Mitigation Framework
    mf = MitigationFramework()
    mf.initialize_mitigation_framework(id2label=id2label,
                                       use_case_name="nurse_vs_all")

    model_name_or_path = "saved_models/nurse_vs_all/bert-base-uncased/best_model/"

    # Load model and tokenizer from disk
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load test data to perform the explanations on
    df = pd.read_csv("saved_datasets/test.csv")
    # Select only the first 300 texts for the sake of time
    texts = df["cleaned_text"].tolist()[:300]

    # Labels to per perform the explanations on (0: non-nurse, 1: nurse)
    label_ids_to_explain = [0, 1]

    # Run the explainer. It returns a dictionary with, for each label id, the list of most important words
    output_dict = mf.run_explainer(model,  # Explained model
                                   tokenizer, # Model Tokenizer
                                   texts,  # Unlabeled corpus of texts to explain and extract most important words
                                   label_ids_to_explain,  # Labels to explain
                                   device=device  # Device to run the explainer on
                                   )

    #Run the identifier to identify the protected attributes from the most important words extracted by the explainer
    df_annotated, protected_attributes, protected_attributes_dict = mf.run_identifier(output_dict,  # Output of the explainer
                                                                                      number_most_important_words=20  # Number of most important words to consider
                                                                                      )

    # Print the protected attributes identified separately for each label
    print(protected_attributes_dict)

    # Load the training dataset (in this case the test dataset for the sake of time) to mitigate
    df_train = pd.read_csv("saved_datasets/test.csv")
    # Select only the first 500 texts for the sake of time
    df_train = df_train.iloc[:500]

    # If this is True, the protected attributes are mitigated separately for each label, otherwise independently of the label
    # For instance, if it is True, the protected attributes identified for the "nurse" label will be used to mitigate only the examples which original label is "nurse" and the same for "non-nurse"
    # If is False, the protected attributes identified for all the labels (e.g., "non-nurse" and "nurse" label) will be used to mitigate all the examples, independently of the original label
    mitigate_each_label_separately = False

    mitigation_strategy = "word_removal"
    #mitigation_strategy = "sentence_removal"

    # Run the moderator to mitigate the protected attributes identified by the identifier in the training dataset
    df_train_mitigated = mf.run_moderator(df_train,  # Training dataset to mitigate
                                          tokenizer,  # Model tokenizer
                                          protected_attributes_dict,  # Protected attributes identified by the identifier
                                          mitigation_strategy=mitigation_strategy,  # Mitigation strategy to use
                                          text_column_name="cleaned_text",  # Name of the column containing the texts
                                          label_column_name="label",  # Name of the column containing the labels
                                          mitigate_each_label_separately=mitigate_each_label_separately,  # Mitigate the protected attributes for each label separately or not
                                          batch_size=128  # Batch size to use for the mitigation
                                          )

    print("Mitigated dataset:")
    print(df_train_mitigated.head(30))

    # Save the mitigated dataset to disk
    df_train_mitigated.to_csv("saved_datasets/test_mitigated.csv", index=False)

    print("End of the mitigation example.")