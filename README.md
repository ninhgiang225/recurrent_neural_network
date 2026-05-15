# GRU Recurrent Neural Network for Text Generation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![RNN](https://img.shields.io/badge/Deep_Learning-Recurrent_Neural_Network-blue?style=for-the-badge)
![GRU](https://img.shields.io/badge/RNN-Gated_Recurrent_Unit-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Character_Level_Model-purple?style=for-the-badge)

## Overview

This project focuses on implementing a **Recurrent Neural Network (RNN)** using a **Gated Recurrent Unit (GRU)** architecture for character-level text generation.

The model is trained on the IMDb movie review dataset and learns to generate synthetic movie reviews character-by-character. By processing text sequentially and maintaining medium-term memory states, the network learns grammatical structure, writing patterns, punctuation usage, and stylistic features from the training data.

This project builds upon bio-inspired concepts of:
- recurrent neural processing,
- temporal sequence learning,
- and memory retention through hidden states.

---

# Project Objectives

- Implement a **Gated Recurrent Unit (GRU)** layer from scratch
- Understand how recurrent neural networks process sequential data
- Explore text generation using character-level NLP models
- Learn how hidden states store medium-term contextual memory
- Train a neural network to generate synthetic movie reviews
- Gain deeper experience with TensorFlow-based sequence models
- Practice vectorized recurrent computation in NumPy and TensorFlow

---

# Project Structure

```bash
.
├── data/
│   ├── little_pigs.csv
│   └── imdb_raw.csv
│
├── notebooks/
│   ├── text_preprocessing.ipynb
│   ├── gru_layer.ipynb
│   └── rnn_text_generation.ipynb
│
├── src/
│   ├── text_dataset_char.py
│   ├── rnn_layers.py
│   └── rnn.py
│
└── README.md
```

---

## Dataset

The project uses two text datasets for development and training.

| File | Description |
|------|-------------|
| `little_pigs.csv` | Small development dataset for debugging |
| `imdb_raw.csv` | Large IMDb movie review dataset used for training |

---

# Character-Level Text Generation

Unlike word-level NLP models, this project operates at the **character level**. The model predicts the next character in a sequence:

```text
Input:
"The movie was amaz"

Target:
"i"
```

Over time, the RNN learns:
- spelling patterns,
- grammar,
- sentence structure,
- punctuation,
- and semantic flow.

---

## Part 1 — Text Preprocessing

### `text_preprocessing.ipynb`

This notebook handles:
- cleaning raw text
- character vocabulary creation
- encoding characters into integers
- sequence generation
- preparing training batches

---

## Part 2 — GRU Layer Implementation

### GRU Components

The GRU layer contains:
- Update Gate: Controls how much previous memory is retained.
- Reset Gate: Controls how much previous information is ignored.
- Candidate Hidden State: Generates new information to store in memory.
- Hidden State: Maintains temporal information across sequences.

---

### `gru_layer.ipynb`

Contains:
- GRU mathematical implementation
- recurrent state updates
- forward propagation logic
- masking support
- testing and debugging

---

## Part 3 — Text Generation RNN

### `rnn.py`

Implements the full recurrent neural network:
- sequence processing
- hidden state propagation
- output prediction
- training loop
- text sampling



### `rnn_layers.py`

Contains custom neural network layers for:
- GRU operations
- dense output layers
- activation functions
- temporal computations



### `text_dataset_char.py`

Responsible for:
- character encoding
- dataset batching
- sequence generation
- train/test preparation

---

## Text Generation Process

After training, the model generates novel text by:

1. Receiving a seed string
2. Predicting the next character
3. Feeding prediction back into the model
4. Repeating recursively

Example:

```text
Seed:
"The movie"

Generated:
"The movie was absolutely incredible and the acting was surprisingly emotional..."
```

---

## Notebooks

### `text_preprocessing.ipynb`

Covers:
- character vocabulary generation
- encoding pipelines
- preprocessing workflow



### `gru_layer.ipynb`

Covers:
- GRU implementation
- recurrent computation
- gating mechanisms
- debugging/tests



### `rnn_text_generation.ipynb`

Covers:
- RNN training
- text generation
- sequence sampling
- generated text analysis

