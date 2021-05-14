Sparse Tensor Classifier
====================================================

.. toctree::
	:hidden:
	:maxdepth: 1
	:caption: Sparse Tensor Classifier

	docs
	tutorials
	references
	contacts
	
Sparse Tensor Classifier (STC) is a supervised classification algorithm for categorical data inspired by the notion of superposition of states in quantum physics. It supports multiclass and multilabel classification, online learning, prior knowledge, automatic dataset balancing, missing data, and provides a native explanation of its predictions both for single instances and for each target class label globally. The algorithm is implemented in SQL and made available via the Python module ``stc`` on `PyPI <https://pypi.org/project/stc/>`_. By default, the library uses an in-memory SQLite database, shipped with Python standard library, that require no configuration by the user. However, it is possible to configure STC to run on alternative DBMS in order to take advantage of persistent storage and scalability.

`> Don't forget to star the repository! <https://github.com/SparseTensorClassifier/stc>`_
	
====================================================
Quickstart
====================================================

Install ``stc`` from `PyPI <https://pypi.org/project/stc/>`_ and make sure to be running ``Python >=3.7``

.. code-block:: python

	pip install stc
	
====================================================
Example
====================================================

Use the Sparse Tensor Classifier to classify animals.
The `dataset <https://www.kaggle.com/uciml/zoo-animal-classification>`_ consists of 101 animals from a zoo.
There are 16 variables with various traits to describe the animals.
The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate
The purpose for this dataset is to be able to predict the classification of the animals.

.. code-block:: python
	
	import pandas as pd
	from stc import SparseTensorClassifier
	
	# Read the dataset
	zoo = pd.read_csv('https://raw.githubusercontent.com/SparseTensorClassifier/tutorial/main/data/zoo/zoo.csv')
	# Initialize the class
	STC = SparseTensorClassifier(targets=['class_type'], features=zoo.columns[1:-1])
	# Fit the training data
	STC.fit(zoo[0:70])
	# Predict the test data
	labels, probability, explainability = STC.predict(zoo[70:])
		
====================================================
Documentation
====================================================
		
Discover the flexibility of the library in the `documentation <docs.html>`_.

====================================================
Tutorials
====================================================

Get started with more advanced `tutorials <tutorials.html>`_.

====================================================
Bug Reports
====================================================

Open an issue on `GitHub <https://github.com/SparseTensorClassifier/stc/issues>`_.

====================================================
Cite as
====================================================

Guidotti E., Ferrara A., (2021). "*An Explainable Probabilistic Classifier for Categorical Data Inspired to Quantum Physics*"

`See references <references.html>`_

====================================================
License
====================================================

All code is released under the `GNU GPL-3 <https://github.com/SparseTensorClassifier/stc/blob/main/LICENSE>`_. If you need to use the same code under different terms, for instance allowing its inclusion in proprietary applications, `contact us <contacts.html>`_ to obtain an `exception <https://www.gnu.org/philosophy/selling-exceptions.html>`_.

