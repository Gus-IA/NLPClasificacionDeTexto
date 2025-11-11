# NLP Sentiment Classification with PyTorch & TorchText

# Clasificación de Sentimientos con PyTorch y TorchText

Este repositorio contiene una implementación moderna en PyTorch para la clasificación de sentimientos usando el dataset IMDB. El modelo es una RNN simple que puede ser opcionalmente bidireccional.

## Resumen

- **Dataset**: Reseñas de películas IMDB (positivas/negativas)  
- **Tokenizer**: Tokenizador en inglés de spaCy (`en_core_web_sm`)  
- **Vocabulario**: Construido a partir del conjunto de entrenamiento, limitado a 10,000 tokens  
- **Modelo**: RNN basada en GRU, con opción bidireccional  
- **Entrenamiento**: Aprendizaje supervisado con función de pérdida cross-entropy  
- **Evaluación**: Precisión (accuracy) calculada sobre el conjunto de prueba  

## Conceptos y técnicas aprendidas

- Carga e iteración del dataset IMDB usando `torchtext.datasets.IMDB`.  
- Tokenización con spaCy y construcción de vocabulario con `torchtext.vocab.build_vocab_from_iterator`.  
- Creación de pipelines para transformar texto y etiquetas en tensores.  
- Manejo de secuencias de longitud variable con `torch.nn.utils.rnn.pad_sequence`.  
- Construcción de `DataLoader` con función `collate` personalizada.  
- Definición de un modelo RNN basado en GRU, incluyendo configuración bidireccional.  
- Ciclo de entrenamiento en PyTorch con cálculo de pérdida y precisión usando `tqdm`.  
- Realización de predicciones sobre nuevas frases.

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
