# NanoGPT-Custom-Training

This repository explores various concepts and techniques in Natural Language Processing (NLP), focusing on **N-Gram Models**, **Markov Processes**, **Text Encoding and Decoding**, and the basics of **Attention Mechanisms**. We dive deep into the mechanics of tokenization and vectorization, understand how models like N-Grams function, and then explore more sophisticated models such as GPT. The code in this repository is inspired by Andrej Karpathy’s work on NLP (link: [Karpathy NLP Lecture Code](https://github.com/karpathy/ng-video-lecture)) and his YouTube video [here](https://www.youtube.com/watch?v=kCc8FmEb1nY).

---

## Introduction

In the field of **Natural Language Processing (NLP)**, the goal is to make machines understand, interpret, and generate human language. Various models and approaches are employed for different tasks, ranging from text classification and translation to text generation.

Here, we will primarily focus on:
- **N-Gram Models**: These are statistical models that predict the likelihood of a word based on the previous 'n-1' words.
- **Markov Processes**: A probabilistic model where future states depend only on the current state, commonly applied to sequence prediction tasks.
- **Text Encoding and Decoding**: Converting text into numerical vectors (tokens) and decoding them back into readable text.
- **Tokenization and Vectorization**: Breaking text into smaller units (tokens), then converting those tokens into vectors for machine learning models.

---

## N-Gram Models and Markov Processes

### What are N-Grams?

An **N-Gram** is a contiguous sequence of 'n' items from a given sample of text or speech. For instance:
- **Unigram**: a single word (e.g., "apple")
- **Bigram**: two consecutive words (e.g., "I love")
- **Trigram**: three consecutive words (e.g., "I love pizza")

These models use the previous 'n-1' words to predict the next word in a sequence, which is particularly useful for text generation tasks.

### Markov Processes

Markov Processes model a sequence of events where the probability of the next event depends only on the current state, not the previous states. This principle is foundational for **N-Gram Models** and is used to predict the next token based on the history of the sequence.

### Text Encoding and Decoding

Text encoding is the process of converting human-readable text into a numerical form that machines can understand. We achieve this through **tokenization** (splitting text into tokens) and **vectorization** (mapping those tokens to vectors).

For example:
- Tokenization: `"I love pizza"` → `["I", "love", "pizza"]`
- Vectorization: `"I" → [1, 0, 0]`, `"love" → [0, 1, 0]`, `"pizza" → [0, 0, 1]`

This process is key for training machine learning models in NLP.

---

## Example of an N-Gram Model

Let’s explore an example of how a **Bigram Model** works. Given a sequence:

```
"I love pizza"
```

- **Unigrams**: ["I", "love", "pizza"]
- **Bigrams**: [("I", "love"), ("love", "pizza")]

The model learns the probabilities of transitioning from one word to the next. For instance, the probability of transitioning from "I" to "love" is higher than from "I" to "pizza".

---

## Environments and Dependency Installation

This repository requires the following dependencies:

- **Python+**
- **PyTorch**
- **NumPy**
- **tqdm**
- **argparse**

To install the dependencies, run the following commands:

```bash
pip install requirements.txt
```

If you're running on a GPU-enabled system, PyTorch will automatically use CUDA for faster training. In case the requirements fails, please manually install each dependency from the code files listed (frozen module package versions weren't checked).

---

## Main Files Breakdown

### 1. `generate_dataset.py`
Run this script first. It installs the Shakespeare Dataset in the `data/` directory, and the filepath will be `data/input.txt` by default. This will be used by every other script.

### 2. `exploration.ipynb`
This notebook explores basic NLP concepts, including tokenization, N-Grams, and attention mechanisms. It walks through the theory and implementation, focusing on simple N-Grams and the fundamentals of attention-based models like Transformers.

### 3. `n_gram_language_model.py` (and `bigram_language_model.py`, `trigram_language_model.py`)
This script implements a basic **N-Gram Language Model**. It demonstrates how to train a model based on N-Grams, such as bigrams or trigrams. It shows how to tokenize, vectorize, and train a model to predict the next word in a sequence using a statistical approach. The setup is the same for the bi-gram and tri-gram versions of the code. The method of execution for the `bigram_language_model.py` and `trigram_language_model.py` is similar to execution of the `n_gram_language_model.py` script.

#### To train:
```bash
python n_gram_language_model.py --mode train
```

#### To generate text:
```bash
python n_gram_language_model.py --mode generate
```

### 4. `gpt.py`
The **GPT Model** implements a simple transformer-based model for text generation. The code implements the architecture from scratch, with key components like multi-head attention and position encoding.

#### Hyperparameters in `gpt.py`:
- **`BATCH_SIZE`**: The number of sequences processed simultaneously.
- **`BLOCK_SIZE`**: The maximum context length for predictions.
- **`MAX_ITERS`**: Total number of training iterations.
- **`LEARNING_RATE`**: Learning rate for the optimizer.
- **`EVAL_INTERVAL`**: How frequently to evaluate the model.
- **`EVAL_ITERS`**: The number of iterations to evaluate on.

#### To train:
```bash
python gpt.py --mode train
```

#### To generate text:
```bash
python gpt.py --mode generate
```

You can tweak the hyperparameters to adjust model performance:
- **Increasing `BLOCK_SIZE`** allows for longer context lengths.
- **Decreasing `LEARNING_RATE`** might stabilize training, especially for larger datasets.
- **Increasing `N_LAYER`** (the number of transformer layers) increases model complexity and learning capacity, but also increases training time.

---

## Understanding `gpt.py`

The **GPT Model** utilizes the Transformer architecture, specifically focusing on the **self-attention mechanism**. It has several key components:

1. **Token Embedding**: Maps tokens to vectors of fixed size.
2. **Positional Encoding**: Injects information about token order into the model.
3. **Transformer Blocks**: Stacks of attention layers for learning context.
4. **Output Layer**: A linear layer that outputs logits for predicting the next token.

### Key Hyperparameters:
- **`N_HEAD`**: Number of attention heads in each attention layer.
- **`N_LAYER`**: Number of transformer blocks stacked on top of each other.
- **`N_EMBEDDING`**: Dimensionality of token embeddings.
- **`DROPOUT`**: Dropout rate to prevent overfitting.
- **`LEARNING_RATE`**: Controls the step size for model updates during training.
- **`BLOCK_SIZE`**: The length of the context window used for predictions.

---

## Conclusion

This repository provides an introduction to NLP techniques such as N-Grams, Markov processes, and transformers, with a focus on hands-on implementation. It explores basic NLP tasks and delves into GPT-style models for text generation.

---

## Acknowledgments

A special thank you to **Andrej Karpathy** for his excellent video lectures and the codebase that inspired much of this work. Check out his lecture series repository [here](https://github.com/karpathy/ng-video-lecture) and watch the video [here](https://www.youtube.com/watch?v=kCc8FmEb1nY) directly.