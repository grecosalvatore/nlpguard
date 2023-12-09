# A Framework for Mitigating the use of Protected Attributes by NLP classifiers

# Table of Contents
- [Mitigation Framework](#mitigation-framework)
- [Setup](#setup)
- [Getting Started](#getting-started)
- [References](#references)
- [Future Developments](#future-developments)



# Mitigation Framework
![Screenshot](images/mitigation-framework-architecture.png)

The mitigation framework that takes an unlabelled corpus of documents, an existing NLP classifier and its training dataset as input to produce a mitigated training
corpus that significantly reduces the learning that takes place on protected attributes without sacrificing
classification accuracy. It does so by means of three components: 
* **Explainer**: detects the most important words used by the classifier to make predictions;
* **Identifier**: detects which of these words are protected attributes;
* **Moderator**: re-trains the classifier to minimize the learning on protected words.

## 1) Explainer
The explainer component leverages XAI techniques to extract the list of most important words used by the model for predictions on the unlabeled corpus.
To this end, it first computes the words' importance within each sentence (local explanation) and then aggregate them across the entire corpus (global explanation).
The framework currently supports the following XAI techniques:
* Integrated Gradients
* SHAP(TODO)

## 2) Identifier
* ChatGPT

## 3) Moderator
* Words Removal
* Sentences Removal

# Setup
1) Create and Start a new environment:
```sh
conda create -n protected-attributes-mitigation-env python=3.8 anaconda
conda activate protected-attributes-mitigation-env
```
2) Install the required packages:
```sh
pip install -r requirements.txt
```

# Getting Started
```python
from framework.mitigation_framework import MitigationFramework
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

id2label = {0: "toxic", 1: "non-toxic"}

model_name_or_path = "your_trained_model_path"

# Load the unlabaled corpus from disk (e.g., test set). Ths is the corpus of texts to explain and extract most important words
df_unalabaled_corpus = pd.read_csv("saved_datasets/unlabeled_corpus.csv")
texts_unlabeled_corpus = df_unalabaled_corpus["text"].tolist()

# Load your model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Instantiate the Mitigation Framework
mf = MitigationFramework().initialize_mitigation_framework(id2label=id2label,
                                                           use_case_name="toxicity-classification")

# Labels to per perform the explanations on (e.g., 0: non-toxic, 1: toxic)
label_ids_to_explain = [0, 1]

# Run the explainer. It returns a dictionary with, for each label id, the list of most important words
output_dict = mf.run_explainer(model,  # Explained model
                               tokenizer, # Model Tokenizer
                               texts_unlabeled_corpus,  # Unlabeled corpus of texts to explain and extract most important words
                               label_ids_to_explain,  # Labels to explain
                               device="cuda:0"  # Device to run the explainer on
                               )

# Identify protected attributes from the 400 most important words extracted by the explainer for each label
number_most_important_words = 400

#Run the identifier to identify the protected attributes from the most important words extracted by the explainer
df_annotated, protected_attributes, protected_attributes_dict = mf.run_identifier(output_dict,  # Output of the explainer
                                                                                  number_most_important_words=number_most_important_words  # Number of most important words to consider
                                                                                  )

# Load the training dataset to mitigate
df_train = pd.read_csv("saved_datasets/test.csv")

# If this is True, the protected attributes are mitigated separately for each label, otherwise independently of the label
# For instance, if it is True, the protected attributes identified for the "nurse" label will be used to mitigate only the examples which original label is "nurse" and the same for "non-nurse"
# If is False, the protected attributes identified for all the labels (e.g., "non-nurse" and "nurse" label) will be used to mitigate all the examples, independently of the original label
mitigate_each_label_separately = False

# Select the mitigation strategy to use
mitigation_strategy = "words_removal"  # Mitigation strategy to use. It can be "words_removal" or "sentences_removal"

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

# Save the mitigated dataset to disk
df_train_mitigated.to_csv("saved_datasets/test_mitigated.csv", index=False)
```

# Future Developments
* Add support for SHAP
* Add support for other XAI techniques

# References
```bibtex

```

# Authors
- Salvatore Greco, *Politecnico di Torino*
- Ke Zhou, *Nokia Bell Labs*
- Licia Capra, *University College London*
- Tania Cerquitelli, *Politecnico di Torino*
- Daniele Quercia, *Nokia Bell Labs*