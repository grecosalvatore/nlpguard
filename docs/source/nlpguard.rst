.. _nlpguard:

NLPGuard
========

NLPGuard is a powerful library for annotating and identifying protected attributes in text data. It consists of three main components: **Explainer**, **Identifier**, and **Moderator**.

This document provides an overview of these components and how they work together.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   explainer
   identifier
   moderator


Explainer
==========

The **Explainer** component extracts the most important predictive words used by the classifier to make predictions.

.. automodule:: nlpguard.explainer.explainer
   :members:
   :undoc-members:
   :show-inheritance:


Identifier
==========

The **Identifier** component determines which of the words extracted by the **Explainer** are protected attributes.

.. automodule:: nlpguard.identifier.identifier
   :members:
   :undoc-members:
   :show-inheritance:


Moderator
==========

The **Moderator** component modifies the original training dataset to produce a new mitigated version to re-train a new mitigated classifier with lower reliance on protected attributes.

.. automodule:: nlpguard.moderator.moderator
   :members:
   :undoc-members:
   :show-inheritance:
