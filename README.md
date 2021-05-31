# Sparse Tensor Classifier

[Sparse Tensor Classifier (STC)](https://sparsetensorclassifier.org/references.html) is a supervised classification algorithm for categorical data inspired by the notion of superposition of states in quantum physics. It supports multiclass and multilabel classification, online learning, prior knowledge, automatic dataset balancing, missing data, and provides a native explanation of its predictions both for single instances and for each target class label globally. Read more at https://arxiv.org/pdf/2105.13988.pdf

The algorithm is implemented in SQL and made available via the Python module ``stc`` on [PyPI](https://pypi.org/project/stc/). By default, the library uses an in-memory SQLite database, shipped with Python standard library, that require no configuration by the user. It is also possible to configure STC to run on [alternative DBMS](https://github.com/SparseTensorClassifier/tutorial/blob/main/Quickstart_DBMS.ipynb) in order to take advantage of persistent storage and scalability.

## Quickstart

Install ``stc`` from [PyPI](https://pypi.org/project/stc/). We recommend using `Python>=3.7` and `SQLite>=3.24.0` for better performance.

```python
pip install stc
```

## Usage

Initialize, fit, and predict. Get started in 3 lines on code! 

**Example**: use Sparse Tensor Classifier to classify animals. The [dataset](https://archive.ics.uci.edu/ml/datasets/Zoo) consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals. The 7 Class Types are: `Mammal`, `Bird`, `Reptile`, `Fish`, `Amphibian`, `Bug` and `Invertebrate`. The purpose for this dataset is to be able to predict the classification of the animals. STC returns a tuple with (1) the predicted classes, (2) the probability for each class, and (3) the contribution of each feature to the target class labels (explainability).

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

Get started with [step-by-step tutorials and use-cases](https://github.com/SparseTensorClassifier/tutorial).

## Cite as

*Guidotti E., Ferrara A., (2021). "An Explainable Probabilistic Classifier for Categorical Data Inspired to Quantum Physics" [arXiv:2105.13988](https://arxiv.org/abs/2105.13988)*

A BibTeX entry for LaTeX users is:

```latex
@misc{stc2021,
  title={An Explainable Probabilistic Classifier for Categorical Data Inspired to Quantum Physics}, 
  author={Emanuele Guidotti and Alfio Ferrara},
  year={2021},
  eprint={2105.13988},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2105.13988}
}
```

___



![](https://raw.githubusercontent.com/SparseTensorClassifier/stc/main/docs/source/_static/img/logo.svg)