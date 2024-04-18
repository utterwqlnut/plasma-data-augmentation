import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np
torch.use_deterministic_algorithms(True)

class PlasmaLSTM(nn.Module):
    # LSTM for binary classification
    def __init__(self, n_input, n_layers, h_size):
        super().__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(n_input, h_size, n_layers, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(h_size),nn.ReLU(),nn.Linear(h_size,1), nn.Sigmoid())

    def forward(self, x):
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_input for i in range(x.shape[0])]
        
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, hidden = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        return self.out(x.mean(dim=1))
    
class PlasmaViewEncoderLSTM(nn.Module):
    def __init__(self, n_input, n_layers, h_size, out_size):
        super().__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(n_input, h_size, n_layers, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(h_size),nn.ReLU(), nn.Linear(h_size,out_size),nn.LayerNorm(out_size)) 
    
    def forward(self, x):
        with torch.no_grad():
            lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_input for i in range(x.shape[0])]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            x, hidden = self.lstm(x)
            x, lengths = pad_packed_sequence(x, batch_first=True)

        mean = x.mean(dim=1)
        out = self.out(mean)
        return out

class TimeSeriesViewMaker(nn.Module):
    # Add noise channel -> Basic Net -> Obtain pertrubation projected onto L1 sphere -> Add pertrubitive noise -> Adversial Loss
    # We only really change this basic net
    # Maybe LSTM or Transformer since it allows for varied time lengths
    def __init__(self, n_dim, n_layers, layer_type, activation, default_distortion_budget, hidden_dim, n_head):
        super().__init__()

        self.n_dim = n_dim
        self.n_layers = n_layers
        self.activation = activation
        self.default_distortion_budget = default_distortion_budget
        
        if layer_type == 'lstm':
            self.net = nn.Sequential(
                nn.LSTM(input_size=self.n_dim+1, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
        elif layer_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(d_model=self.n_dim+1, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            self.net = nn.Sequential(
                nn.TransformerEncoder(transformer_layer,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )

    def add_noise_channel(self, x):
        return torch.cat((x,torch.rand(x[:,:,0].unsqueeze(-1).shape)),dim=-1)
    
    def get_delta(self, y_pixels, specified_distortion_budget=None, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere taken'''
        # Taken from viewmaker github
        
        distortion_budget =  self.default_distortion_budget if specified_distortion_budget==None else specified_distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]

        avg_magnitude = delta.abs().mean([1,2], keepdim=True)

        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, specified_distortion_budget=None):
        out = self.add_noise_channel(x)
        out = self.net(out)

        out = self.get_delta(out,specified_distortion_budget)

        out = x+out
        return out
    
class extract_tensor(nn.Module):
    def forward(self,x):
        return x[0]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series From DLinear
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block From DLinear
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class DecompTimeSeriesViewMaker(nn.Module):
    def __init__(self, n_dim, n_layers, layer_type, activation, default_distortion_budget, hidden_dim, n_head):
        super().__init__()

        self.n_dim = n_dim
        self.n_layers = n_layers
        self.activation = activation
        self.default_distortion_budget = default_distortion_budget
        self.decomp = series_decomp(25)
        self.conv1 = nn.ConvTranspose1d(self.n_dim+1,32,kernel_size=9)
        
        if layer_type == 'lstm':
            self.netT = nn.Sequential(
                nn.LSTM(input_size=self.n_dim+1, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
            self.netS = nn.Sequential(
                nn.LSTM(input_size=self.n_dim+1, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
        elif layer_type == 'transformer':
            transformer_layer = nn.TransformerEncoderLayer(d_model=self.n_dim+1, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            self.netT = nn.Sequential(
                nn.TransformerEncoder(transformer_layer,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
            self.netS = nn.Sequential(
                nn.TransformerEncoder(transformer_layer,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )

    def add_noise_channel(self, x):
        return torch.cat((x,torch.rand(x[:,:,0].unsqueeze(-1).shape)),dim=-1)
    
    def get_delta(self, y_pixels, specified_distortion_budget=None, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere taken'''
        # Taken from viewmaker github
        
        distortion_budget =  self.default_distortion_budget if specified_distortion_budget==None else specified_distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]

        avg_magnitude = delta.abs().mean([1,2], keepdim=True)

        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, specified_distortion_budget=None):
        out = self.add_noise_channel(x)

        seasonal, trend = self.decomp(out)
        out_s = self.netS(seasonal)
        out_t = self.netT(trend)
        combined = out_s+out_t

        delta = self.get_delta(combined,specified_distortion_budget)

        out = x+delta
        return out
    
class extract_tensor(nn.Module):
    def forward(self,x):
        return x[0] 