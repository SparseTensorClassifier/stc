# Sparse Tensor Classifier

Sparse Tensor Classifier (STC) is a supervised classification algorithm for categorical data inspired by the notion of superposition of states in quantum physics. It supports multiclass and multilabel classification, online learning, prior knowledge, automatic dataset balancing, missing data, and provides a native explanation of its predictions both for single instances and for each target class label globally. 

The algorithm is implemented in SQL and made available via the Python module ``stc`` on [PyPI](https://pypi.org/project/stc/). By default, the library uses an in-memory SQLite database, shipped with Python standard library, that require no configuration by the user. However, it is possible to configure STC to run on alternative DBMS in order to take advantage of persistent storage and scalability.

## Quickstart

Install ``stc`` from [PyPI](https://pypi.org/project/stc/) and make sure to be running ``Python >=3.7``

```python
pip install stc
```

## Example

Use the Sparse Tensor Classifier to classify animals. The [dataset](https://www.kaggle.com/uciml/zoo-animal-classification) consists of 101 animals from a zoo.
There are 16 variables with various traits to describe the animals. The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate. The purpose for this dataset is to be able to predict the classification of the animals.

```python
import pandas as pd
from stc import SparseTensorClassifier

# Read the dataset
zoo = pd.read_csv('https://git.io/Jss6f')
# Initialize the class
STC = SparseTensorClassifier(targets=['class_type'], features=zoo.columns[1:-1])
# Fit the training data
STC.fit(zoo[0:70])
# Predict the test data
labels, probability, explainability = STC.predict(zoo[70:])
```

## Documentation

Discover the flexibility of the library in the [documentation](https://sparsetensorclassifier.org/docs.html).

## Tutorials

Get started with more advanced [tutorials](https://github.com/SparseTensorClassifier/tutorial).

## Cite as

```latex
@Misc{stc2021,
  title = {An Explainable Probabilistic Classifier for Categorical Data Inspired to Quantum Physics},
  author = {Autori...},
  year = {2021},
  eprint = {arXiv:2101.00086},
  url = {https://arxiv.org/abs/...}
}
```

___



![](./docs/source/_static/img/logo.svg)