# NLP Sentiment Classification with PyTorch & TorchText

This repository contains a modern PyTorch implementation of sentiment classification using the IMDB dataset. The model is a simple RNN that can optionally be bidirectional.

## Overview

- **Dataset**: IMDB movie reviews (positive/negative)
- **Tokenizer**: spaCy English tokenizer (`en_core_web_sm`)
- **Vocabulary**: Built from training set, limited to 10,000 tokens
- **Model**: GRU-based RNN, with optional bidirectional architecture
- **Training**: Standard supervised training with cross-entropy loss
- **Evaluation**: Accuracy computed on the test split

## Features Learned

- Loading and iterating over IMDB dataset with `torchtext.datasets.IMDB`.
- Tokenization with spaCy and building a vocabulary with `torchtext.vocab.build_vocab_from_iterator`.
- Creating pipelines for text and labels.
- Handling variable-length sequences using `torch.nn.utils.rnn.pad_sequence`.
- Building `DataLoader` with a custom collate function.
- Defining a GRU-based RNN model, including bidirectional configuration.
- Training loop with PyTorch, including tracking loss and accuracy using `tqdm`.
- Making predictions on new sentences.

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
