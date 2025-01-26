.. _nlpguard:

NLPGuard
========

NLPGuard is a powerful library for annotating and identifying protected attributes in text data. It consists of three main components: **Explainer**, **Identifier**, and **Moderator**.

This document provides an overview of these components and their role in improving fairness and interpretability in NLP models.




.. toctree::
   :maxdepth: 2
   :caption: NLPGuard Components:

   explainer/index
   identifier/index
   moderator/index

.. automodule:: nlpguard
   :members:
   :undoc-members:
   :show-inheritance:



Explainer
=========

The **Explainer** component, part of the `nlpguard` library, extracts the most important predictive words used by the classifier to make predictions.

Overview
--------

The **Explainer** identifies influential features contributing to the model's decisions, enabling users to interpret and debug the behavior of NLP models.

For more details, refer to:
- :doc:`Explainer Overview <explainer/index>`
- :doc:`API Documentation <explainer/api>`


Identifier
==========

The **Identifier** component, part of the `nlpguard` library, determines which of the words extracted by the **Explainer** are protected attributes.

Overview
--------

The **Identifier** flags features or attributes that are sensitive or protected, such as those related to race, gender, or age, ensuring that potential bias in models can be addressed.

For more details, refer to:
- :doc:`Identifier Overview <identifier/index>`
- :doc:`API Documentation <identifier/api>`


Moderator
=========

The **Moderator** component, part of the `nlpguard` library, modifies the original training dataset to produce a new mitigated version that reduces reliance on protected attributes.

Overview
--------

The **Moderator** helps ensure fairness by mitigating the model’s dependency on sensitive features identified by the **Identifier**.

For more details, refer to:
- :doc:`Moderator Overview <moderator/index>`
- :doc:`API Documentation <moderator/api>`
