"""Model definition + preprocessing, used by app.py."""
import torch
import torch.nn as nn

MAX_LENGTH = 50


def text_to_numbers(text, word_to_number):
    numbers = []
    for word in text.lower().split()[:MAX_LENGTH]:
        numbers.append(word_to_number.get(word, word_to_number["<UNK>"]))
    while len(numbers) < MAX_LENGTH:
        numbers.append(word_to_number["<PAD>"])
    return numbers


class LSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=300, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_size,
            batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).cpu().clamp(min=1)
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(summary))
