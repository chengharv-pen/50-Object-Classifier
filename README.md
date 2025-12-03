# Team-26

Intended to run locally, in your IDE of choice.

We provide the saved models in the `./models` directory so that there would be no need to retrain the models.

The <a href="https://drive.google.com/drive/folders/1zpvh664rCQgniZHCO7pZlFxOlheQJdqt?usp=sharing">datasets</a> (`test.csv`, `train.csv`) must be placed in the data directory, since GitHub blocks files larger than 100 MiB (<a href="https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github">see this</a>).

![Data Directory](images/data_directory.png)

Given the following system, the model takes around 11 minutes to train on the full train dataset:
```
IDE: PyCharm
Python Interpreter: Miniconda (conda version 25.9.1)

Operating System: Linux Mint 22.2 Cinnamon
Cinnamon Version: 6.4.8
Linux Kernel: 6.8.0-85-generic
Processor: AMD Ryzen 7 5825U with Radeon Graphics × 8
Memory: 15.0 GiB
Graphics Card: NVIDIA Corporation GA107M [GeForce RTX 3050 Ti Mobile]
```

We provide a `requirements.txt` file for pip, but using it may not work due to CUDA libraries compatibility. See the installed pip packages below.

Installed pip packages are:

```
numpy
pandas
matplotlib
seaborn
notebook
ipywidgets
scikit-learn
torch
tqdm
```
# Purpose of each file

### Root Directory

`COMP432 Report.pdf`

- The project report

`MediumMLP-RandomSearch.ipynb`

- Performs Random Search on a hyperparameter search space.

`MediumMLP-KFoldValidation.ipynb`

- Performs K-Fold Validation using a set of hyperparameters obtained from Random Search.

`MediumMLP-VerifyValAcc`

- Train (or load) the model on the training and validation subsets using a set of hyperparameters. Can save the model. This file also contains the report's visualizations.

`MediumMLP-EvaluateTest`

- Train (or load) the model on the full training dataset using a set of hyperparameters. Can save the model. This file generates the CSV and JSON files in the Submission Directory.

### Submissions Directory

CSV files used for Kaggle submissions, along with JSON files that contain the model's hyperparameters. Timestamps are include in the file names to make submission tracking easier.

### Models Directory

Where the saved models are located.

### Data Directory

This directory is where `test.csv` and `train.csv` should be placed in.

### Modules Directory

`__init__.py`

- Makes the Python files in this directory accessible to the notebooks for importing.

`medium_mlp.py`

- Defines the Object Classifier model.

`utils.py`

- The core of the project. Contains a Utils class that has methods to set seed, save and load models, train the model on training and validation subsets, and train the model on full training dataset.

### Images Directory

Contains visualizations from `MediumMLP-VerifyValAcc`

# Specific

There may be a need to install the GPU version of `torch` in Windows 11. For example, with a GTX 1060 (CUDA Version 12.2):

`pip install torch --index-url https://download.pytorch.org/whl/cu121`

Try `nvidia-smi` to see your GPU's CUDA Version (if it is an NVIDIA GPU).
