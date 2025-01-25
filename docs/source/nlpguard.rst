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

The **Explainer** component of NLPGuard is responsible for explaining how different tokens are categorized into protected attributes. This module is useful when you need to understand why a certain token was classified under a particular protected attribute.

.. automodule:: nlpguard.explainer.explainer
   :members:
   :undoc-members:
   :show-inheritance:


Identifier
==========

The **Identifier** component focuses on identifying tokens in a text as protected attributes using various models, such as ChatGPT or LLama. It provides methods for annotating tokens with labels, such as "Age", "Disability", etc.

.. automodule:: nlpguard.identifier.identifier
   :members:
   :undoc-members:
   :show-inheritance:


Moderator
==========

The **Moderator** component works in conjunction with the Explainer and Identifier modules to moderate and ensure that identified protected attributes are not used inappropriately within automated decision-making systems.

.. automodule:: nlpguard.moderator.moderator
   :members:
   :undoc-members:
   :show-inheritance:
