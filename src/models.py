import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np

class PlasmaLSTM(nn.Module):
    # LSTM for binary classification
    def __init__(self, n_input, n_layers, h_size):
        super().__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(n_input, h_size, n_layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h_size,1),nn.Sigmoid())

    def forward(self, x):
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_input for i in range(x.shape[0])]
        
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, hidden = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        return self.out(x.mean(dim=1))
    
class PlasmaViewEncoderLSTM(nn.Module):
    def __init__(self, n_input, n_layers, h_size):
        super().__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(n_input, h_size, n_layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(h_size,n_input),nn.ReLU(),nn.Linear(n_input,n_input)) 
    
    def forward(self, x):
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_input for i in range(x.shape[0])]
        
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, hidden = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        return self.out(x.mean(dim=1))

class TimeSeriesViewMaker(nn.Module):
    # Add noise channel -> Basic Net -> Obtain pertrubation projected onto L1 sphere -> Add pertrubitive noise -> Adversial Loss
    # We only really change this basic net
    # Maybe LSTM or Transformer since it allows for varied time lengths
    def __init__(self, n_dim, n_layers, layer_type, activation, distortion_budget, hidden_dim, n_head):
        super().__init__()

        self.n_dim = n_dim
        self.n_layers = n_layers
        self.activation = activation
        self.distortion_budget = distortion_budget

        if layer_type == 'cnn':
            # Traditional Viewmaker with 1d CNNs
            pass
        elif layer_type == 'lstm':
            self.net = nn.Sequential(
                nn.LSTM(input_size=self.n_dim+1, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
        elif layer_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(d_model=self.n_dim+1, n_head=n_head, dim_feedforward=hidden_dim, batch_first=True)
            self.net = nn.Sequential(
                nn.TransformerEncoder(transformer_layer,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )

    def add_noise_channel(self, x):
        return torch.cat((x,torch.rand(x[:,:,0].unsqueeze(-1).shape)),dim=-1)
    
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere taken'''
        # Taken from viewmaker github
        
        distortion_budget = self.distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]

        avg_magnitude = delta.abs().mean([1,2], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x):
        out = self.add_noise_channel(x)
        out = self.net(out)
        out = self.get_delta(out)

        out = x+out
        return out
    
class extract_tensor(nn.Module):
    def forward(self,x):
        return x[0]