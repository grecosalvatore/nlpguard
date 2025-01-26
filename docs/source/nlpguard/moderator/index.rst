Moderator
=========

The **Explainer** component extracts the most important predictive words used by the classifier to make predictions. It is designed to help interpret the outputs of machine learning models by identifying influential features.

Overview
--------

The **Explainer** works by:
- Analyzing the model's predictions and the features that contribute most to them.
- Highlighting the top predictive words for each prediction.

This tool is particularly useful for:
- Debugging model behavior.
- Ensuring transparency and interpretability of NLP models.

### Key Features
- Extraction of top predictive words.
- Support for multiple classifier types.
- Easy integration with popular NLP pipelines.

### Example Usage

Here’s an example of how to use the **Explainer** component:

.. code-block:: python

   from nlpguard.explainer import Explainer

   # Instantiate the Explainer
   explainer = Explainer(model=my_model, tokenizer=my_tokenizer)

   # Explain a prediction
   prediction_explanation = explainer.explain("The text to analyze")

   print(prediction_explanation)

For more details on the available methods, refer to the **API Documentation** below.

API Documentation
-----------------
For a complete reference of the available methods and classes, check the :doc:`api`.

.. toctree::
   :maxdepth: 2
   :caption: Moderator Documentation:
   api