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
# References
```bibtex

```