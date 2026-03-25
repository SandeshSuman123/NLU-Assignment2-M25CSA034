# NLU-Assignment2-M25CSA034

This repository contains the implementation of Assignment 2 for Natural Language Understanding.

The assignment is divided into two parts:
- Problem 1: Word2Vec (CBOW and Skip-gram)
- Problem 2: Character-level Name Generation using RNNs

---

## Repository Structure

### Problem1
- task1_dataset_preprocessing.py
- task2_train_models.py
- task3_semantic_analysis.py
- task4_visualize.py

### Problem2
- train_models.py
- generate_names.py
- evaluate_models.py

---

## Requirements

Python 3.8 or above is required.

Install dependencies using:

pip install numpy matplotlib scikit-learn torch

---

## How to Run

### Problem 1

Go to the folder:

cd Problem1

Run preprocessing:

python task1_dataset_preprocessing.py

Train CBOW and Skip-gram models:

python task2_train_models.py

Run semantic analysis:

python task3_semantic_analysis.py

Run visualization:

python task4_visualize.py

---

### Problem 2

Go to the folder:

cd Problem2

Train models (RNN, BLSTM, Attention):

python train_models.py

Generate names:

python generate_names.py

Evaluate models:

python evaluate_models.py

---



## Details

- CBOW and Skip-gram models are implemented from scratch
- RNN, LSTM models are implemented using PyTorch
- Evaluation includes novelty rate and diversity
- Visualization includes PCA and t-SNE plots

---

## Author
Sandesh Suman


