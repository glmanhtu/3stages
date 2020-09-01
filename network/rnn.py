import torch
from torch import nn

from utils import dl_utils
from utils.constants import device


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_classes < 2:
            dropout = 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.logits = nn.Linear(hidden_size, num_classes)
        self.apply(dl_utils.weights_init_xavie)

    def forward(self, sequences):
        h0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(sequences, (h0, c0))

        # Decode the hidden state of the last time step
        x = self.logits(out[:, -1, :])
        return x.view(-1)

