import random
import torch
import torch.nn as nn 
from sklearn.metrics import confusion_matrix
from typing import Any, Callable, List, Optional, Type, Union

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.wz = nn.Linear(input_size, hidden_size)
        self.uz = nn.Linear(hidden_size, hidden_size)
        self.wr = nn.Linear(input_size, hidden_size)
        self.ur = nn.Linear(hidden_size, hidden_size)
        self.wh = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        z = self.sigmoid(self.wz(x) + self.uz(h))
        r = self.sigmoid(self.wr(x) + self.ur(h))
        h_tild = self.tanh(self.wh(x) + self.uh(r * h))
        h_new = (1-z) * h + z * h_tild
        return h_new, h_new # we use the same thing for the output
        

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = CustomGRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        """Take in a sequence, process it directly"""
        hidden = self._init_hidden().to(x.device)
        for k in range(x.shape[0]):
            _, hidden = self.gru(x[k], hidden)
        out = self.fc(self.relu(hidden))
        out = out.unsqueeze(0)
        out = self.logsoftmax(out)
        return out

    def _init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.hidden_size).zero_()
        return hidden
    

class RNNWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Take in a sequence, process it directly"""
        hidden = self._init_hidden().to(x.device)
        out, hidden_n = self.gru(x, hidden)
        out = self.fc(self.relu(hidden_n[-1,0,:]))
        # print(out.shape)
        out = out.unsqueeze(0)
        out = self.logsoftmax(out)
        return out

    def _init_hidden(self):
        weight = next(self.parameters()).data
        d = 2 if self.bidirectional else 1
        hidden = weight.new(d *  self.num_layers,          1,         self.hidden_size    ).zero_()        
        return hidden


class GRUnetWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1): 

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(  self.num_layers,          1,         self.hidden_size    ).zero_()
        return hidden