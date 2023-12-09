# A Framework for Mitigating the use of Protected Attributes by NLP classifiers

# Table of Contents
- [Setup](#setup)
- [Mitigation Framework](#mitigation-framework)
- [References](#references)

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
* [Integrated Gradients]()
* [SHAP(TODO)]()

## 2) Identifier
* [ChatGPT]()

## 3) Moderator
* [Words Removal]()
* [Sentences Removal]()

# References
```bibtex

```

# Authors
- Salvatore Greco
- Ke Zhou
- Licia Capra
- Tania Cerquitelli
- Daniele Quercia