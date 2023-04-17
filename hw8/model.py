import random
import torch
import torch.nn as nn 
from sklearn.metrics import confusion_matrix
from typing import Any, Callable, List, Optional, Type, Union

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.wz = nn.Linear(input_size, hidden_size)
        self.uz = nn.Linear(hidden_size, hidden_size)
        self.wr = nn.Linear(input_size, hidden_size)
        self.wh = nn.Linear(hidden_size, hidden_size)
        self.wh = nn.Linear(input_size, hidden_size)
        self.uh = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        z = self.sigmoid(self.wz(x) + self.uz(h))
        r = self.sigmoid(self.wr(x) + self.ur(h))
        h_tild = self.tanh(self.wh(x) + self.uh(r * h))
        h_new = (1-z) * h + z * h_tild

        return 
        

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, h):
        pass


class GRUnetWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
        """
        -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                embedding, the value of this variable will always be equal to 300.
        -- hidden_size is the size of the hidden state in the RNN
        -- output_size is the size of output of the RNN.  For binary classification of 
                input text, output_size is 2.
        -- num_layers creates a stack of GRUs
        """
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
        #                  num_layers  batch_size    hidden_size
        # hidden = weight.new(  2,          1,         self.hidden_size    ).zero_()
        hidden = weight.new(  self.num_layers,          1,         self.hidden_size    ).zero_()
        return hidden