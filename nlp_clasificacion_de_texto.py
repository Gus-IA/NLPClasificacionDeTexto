import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
import spacy

# -----------------------------
# CONFIGURACIÓN BÁSICA
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")


# -----------------------------
# DESCARGA Y PROCESAMIENTO DEL DATASET
# -----------------------------
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)


train_iter, test_iter = IMDB(split=("train", "test"))

# Crear vocabulario (solo con entrenamiento)
vocab = build_vocab_from_iterator(
    yield_tokens(IMDB(split="train")), max_tokens=10000, specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])

print("Tamaño del vocabulario:", len(vocab))


# -----------------------------
# PIPELINES DE TEXTO Y ETIQUETA
# -----------------------------
def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return 1 if x == "pos" else 0


# -----------------------------
# FUNCIÓN COLLATE PARA DATALOADER
# -----------------------------
def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        texts.append(torch.tensor(text_pipeline(text), dtype=torch.int64))
    texts = pad_sequence(texts, batch_first=False, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.int64)
    return texts.to(device), labels.to(device)


train_dataloader = DataLoader(
    list(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    list(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch
)

dataloader = {"train": train_dataloader, "test": test_dataloader}


# -----------------------------
# DEFINICIÓN DEL MODELO RNN
# -----------------------------
class RNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim=128,
        hidden_dim=128,
        output_dim=2,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(
            2 * hidden_dim if bidirectional else hidden_dim, output_dim
        )

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        return self.fc(hidden)


# -----------------------------
# ENTRENAMIENTO
# -----------------------------
def fit(model, dataloader, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader["train"], desc=f"Epoch {epoch}/{epochs}")
        for X, y in bar:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            acc = (y_hat.argmax(1) == y).float().mean().item()
            train_loss.append(loss.item())
            train_acc.append(acc)
            bar.set_postfix(loss=np.mean(train_loss), acc=np.mean(train_acc))

        # Validación
        model.eval()
        val_loss, val_acc = [], []
        with torch.no_grad():
            for X, y in dataloader["test"]:
                y_hat = model(X)
                loss = criterion(y_hat, y)
                acc = (y_hat.argmax(1) == y).float().mean().item()
                val_loss.append(loss.item())
                val_acc.append(acc)
        print(
            f"Epoch {epoch}: loss={np.mean(train_loss):.4f}, "
            f"val_loss={np.mean(val_loss):.4f}, "
            f"acc={np.mean(train_acc):.4f}, val_acc={np.mean(val_acc):.4f}"
        )


# -----------------------------
# EJECUCIÓN
# -----------------------------
model = RNN(input_dim=len(vocab))
fit(model, dataloader)

# -----------------------------
# PREDICCIÓN
# -----------------------------
nlp = spacy.load("en_core_web_sm")


def predict_sentence(model, sentence):
    model.eval()
    tokens = tokenizer(sentence)
    ids = [vocab[token] for token in tokens]
    X = torch.tensor(ids, dtype=torch.int64).unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(X)
        label = torch.argmax(pred, dim=1).item()
    return "pos" if label == 1 else "neg"


sentences = [
    "this film is terrible",
    "this film is great",
    "this film is good",
    "a waste of time",
]

for s in sentences:
    print(f"{s!r} → {predict_sentence(model, s)}")

# -----------------------------
# ENTRENAR MODELO BIDIRECCIONAL (opcional)
# -----------------------------
model_bi = RNN(input_dim=len(vocab), bidirectional=True)
fit(model_bi, dataloader)
