# Quantum Machine Learning for Time Series Forecasting

This repository presents a hybrid quantum-classical machine learning approach to time series forecasting, integrating classical machine learning techniques with quantum circuits. This project aims to explore the potential of quantum machine learning (QML) in predictive analytics, offering a novel solution for complex forecasting challenges.

Currently, the project works with two models: the classical [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model and its hybrid quantum-classical analog called HQCLSTM from [this paper](https://arxiv.org/abs/2312.16379); the models are compared on [the AAPL Stock dataset](https://huggingface.co/datasets/chrisaydat/applestockpricehistory).

## Table of Contents

- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Main Notebook](#main-notebook)
- [Dependencies](#dependencies)
- [License](#license)

## Repository Structure

```bash
├── data/  # Datasets used in the project
├── evaluation/  # Saved results of model evaluation
├── models/  # Saved trained models
├── src/  # Source code
│   └── qml_time_series.ipynb  # Main Jupyter notebook
├── training/  # Saved results of model training
├── LICENSE.md  # License file
└── README.md  # Project README file
```

## Getting Started

To simply browse the project, you can view the main Jupyter notebook [here on GitHub](src/qml_time_series.ipynb), on [Google Colab](https://colab.research.google.com/github/Ivan-Sergeyev/qml-time-series/blob/main/src/qml_time_series.ipynb), or locally.

To run and edit the project, the easiest way to get started is to import the repository to [Google Colab](https://colab.research.google.com/) following these steps.

1. **Get a copy of the project:** `git clone` it or download and unpack the .zip archive.
1. **Save the project to your [Google Drive](https://drive.google.com):**
   1. **Using the web interface:** use the 'folder upload' function.
   1. **Using the app:** save the project to a folder that is synced with your drive.
1. **Open the main Jupyter notebook on [Google Colab](https://colab.research.google.com/):**
   1. Open the 'Recent' or 'Google Drive' tab
   1. Search for `qml_time_series.ipynb`
   1. Open the notebook.

Colab is the preferred option because of how importing the ML packages works and how much computational power is needed to train the models, especially the quantum ones. Running everything locally may be possible, but I have not tested it and cannot offer any support in case you try to do it.

## Main Notebook

The core of this project is in [the main Jupyter notebook](src/qml_time_series.ipynb). It covers data loading, preprocessing, model architecture (both classical and quantum), training, evaluation, and result visualization.

## Dependencies

All prerequisite libraries are installed at the beginning of the notebook, most notably including:

- [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) for working with classical ML models
- [TensorFlow Quantum](https://www.tensorflow.org/quantum) for working with quantum ML models
- [Cirq](https://quantumai.google/cirq) for working with quantum circuits
- [SymPy](https://docs.sympy.org/latest/index.html) for handling symbolic math
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) for storing and analyzing datasets
- [pickle](https://docs.python.org/3/library/pickle.html) for serializing data
- [seaborn](https://seaborn.pydata.org/), [matplotlib](https://matplotlib.org/) for visualizing data

Colab may require you to restart the session and re-run the notebook to make sure all dependencies are installed properly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
