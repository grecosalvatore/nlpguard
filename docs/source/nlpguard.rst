.. _nlpguard:

NLPGuard
==================

The NLPGuard framework is designed for mitigating protected attributes in text classification tasks. It is composed of three main components: Explainer, Identifier, and Moderator.

.. toctree::
   :maxdepth: 2
   :caption: Components

   explainer
   identifier
   moderator

Explainer Component
==================

The Explainer component is responsible for extracting important words from a corpus of text, helping to explain model predictions. Methods such as Integrated Gradients and SHAP can be used.

.. automodule:: nlpguard.explainer
   :members:
   :undoc-members:
   :show-inheritance:

Identifier Component
===================

The Identifier component identifies protected attributes from the most important words extracted by the Explainer.

.. automodule:: nlpguard.identifier
   :members:
   :undoc-members:
   :show-inheritance:

Moderator Component
==================

The Moderator component mitigates the identified protected attributes in a corpus of text using various strategies such as word removal or word replacement.

.. automodule:: nlpguard.moderator
   :members:
   :undoc-members:
   :show-inheritance:
