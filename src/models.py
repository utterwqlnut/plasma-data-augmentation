import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np
import math
torch.manual_seed(42)

class LSTMFormer(nn.Module):
    def __init__(self,
                 buffer_size=5,
                 input_size=12,
                 embedding_dim=128,
                 n_lstm_layers=1,
                 lstm_dropout=0.26251692966017,
                 bidirectional=False,
                 n_head=2,
                 n_inner=512,
                 transformer_dropout=0.16762577671904463,
                 n_layers=4,
                 ):

            super().__init__()
            self.input_size = input_size

            self.buffer_size = buffer_size

            self.buffer = [0.1]*buffer_size

            self.lstm = nn.LSTM(input_size, embedding_dim, n_lstm_layers,dropout=lstm_dropout,bidirectional=bidirectional, batch_first=True)

            self.norm = nn.LayerNorm(embedding_dim)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=n_inner, batch_first=True, dropout=transformer_dropout)

            self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

            self.fc = nn.Sequential(nn.Linear(embedding_dim,1),nn.Sigmoid())

            self.fc_pretrain = nn.Linear(embedding_dim, input_size)

            self.mode = "normal"

    def forward(self,
            inputs_embeds,
            plot_attentions=False,
            need_buffer=False,
            **kwargs):

        # Learnable positional encodings LSTM style
        x = inputs_embeds.float()

        lengths = [len(inputs_embeds[i])-len(torch.where(inputs_embeds[i]==-100)[0])/self.input_size for i in range(inputs_embeds.shape[0])]

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, hidden = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        x = self.norm(x)

        # Encoder
        if plot_attentions:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1),device=self.device)

            att_w = []
            for i in range(len(self.encoder.layers)):
                att_w.append(self.encoder.layers[i].self_attn(x,x,x,attn_mask=tgt_mask,need_weights=True)[1])

            att_w = torch.stack(att_w)
        else:
            att_w = None

        x = self.encoder(x)

        if self.mode == "normal":
            x = self.fc(x.mean(dim=1))
        else:
            x = self.fc_pretrain(x)

        if need_buffer:
            self.buffer = self.buffer[1:]
            self.buffer.append(x)
            x = sum(self.buffer)/self.buffer_size

        return x
    def reset_buffer(self):
        self.buffer = [0.1]*self.buffer_size

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
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_input for i in range(x.shape[0])]
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        x, hidden = self.lstm(x)
        x, lengths = pad_packed_sequence(x, batch_first=True)

        mean = x.mean(dim=1)
        out = self.out(mean)
        return out

class PlasmaViewEncoderTransformer(nn.Module):
    def __init__(self, n_input, n_layers, h_size, out_size):
        super().__init__()
        self.n_input = n_input
        self.n_layers = n_layers
        self.h_size = h_size
        self.transformer_layer = nn.TransformerEncoderLayer(n_input,1,h_size, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer,n_layers)
        self.out = nn.Sequential(nn.LayerNorm(n_input),nn.ReLU(), nn.Linear(n_input,out_size),nn.LayerNorm(out_size))
        self.encoder = PositionalEncoding(12,max_len=2048)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
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
        return torch.cat((x,torch.rand(x[:,:,0].unsqueeze(-1).shape, device=x.device)),dim=-1)

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

        out = x + out
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
            transformer_layer_1 = nn.TransformerEncoderLayer(d_model=self.n_dim+1, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=self.n_dim+2, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            transformer_layer_3 = nn.TransformerEncoderLayer(d_model=self.n_dim+3, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)

            self.netT = nn.Sequential(
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_1,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_2,n_layers),
                nn.LayerNorm(self.n_dim+2),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_3,n_layers),
                nn.LayerNorm(self.n_dim+3),
                activation,
                nn.Linear(self.n_dim+3, self.n_dim),
            )
            self.netS = nn.Sequential(
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_1,n_layers),
                nn.LayerNorm(self.n_dim+1),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_2,n_layers),
                nn.LayerNorm(self.n_dim+2),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_3,n_layers),
                nn.LayerNorm(self.n_dim+3),
                activation,
                nn.Linear(self.n_dim+3, self.n_dim),
            )


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
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_dim for i in range(x.shape[0])]

        seasonal, trend = self.decomp(x)

        out_s = self.netS(seasonal)
        out_t = self.netT(trend)
        combined = out_s+out_t

        delta = self.get_delta(combined,specified_distortion_budget)

        out = x+delta

        for i,length in enumerate(lengths):
            out[i,int(length):, ...] = -100

        return out

class DecompTimeSeriesViewMakerConv(nn.Module):
    def __init__(self, n_dim, n_layers, layer_type, activation, default_distortion_budget, hidden_dim, n_head):
        super().__init__()

        self.n_dim = n_dim
        self.n_layers = n_layers
        self.activation = activation
        self.default_distortion_budget = default_distortion_budget
        self.decomp = series_decomp(25)

        if layer_type == 'lstm':
            self.netT = nn.Sequential(
                transpose(),
                nn.Conv1d(self.n_dim+1,64,3),
                transpose(),
                nn.LayerNorm(64),
                activation,
                nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,64),
                nn.LayerNorm(64),
                transpose(),
                nn.ConvTranspose1d(64,self.n_dim+1,3),
                transpose(),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
            self.netS = nn.Sequential(
                transpose(),
                nn.Conv1d(self.n_dim+1,64,3),
                transpose(),
                nn.LayerNorm(64),
                activation,
                nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True),
                extract_tensor(),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim,64),
                nn.LayerNorm(64),
                transpose(),
                nn.ConvTranspose1d(64,self.n_dim+1,3),
                transpose(),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1,self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
        elif layer_type == 'transformer':
            transformer_layer_1 = nn.TransformerEncoderLayer(d_model=64+1, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=64+2, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)
            transformer_layer_3 = nn.TransformerEncoderLayer(d_model=64+3, nhead=n_head, dim_feedforward=hidden_dim, batch_first=True)

            self.netT = nn.Sequential(
                add_noise_channel(),
                transpose(),
                nn.Conv1d(self.n_dim+1,64,3),
                transpose(),
                nn.LayerNorm(64),
                activation,
                #PositionalEncoding(d_model=64,max_len=2048),
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_1,n_layers),
                nn.LayerNorm(64+1),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_2,n_layers),
                nn.LayerNorm(64+2),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_3,n_layers),
                nn.LayerNorm(64+3),
                activation,
                transpose(),
                nn.ConvTranspose1d(64+3,self.n_dim+1,3),
                transpose(),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )
            self.netS = nn.Sequential(
                add_noise_channel(),
                transpose(),
                nn.Conv1d(self.n_dim+1,64,3),
                transpose(),
                nn.LayerNorm(64),
                activation,
                #PositionalEncoding(d_model=64,max_len=2048),
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_1,n_layers),
                nn.LayerNorm(64+1),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_2,n_layers),
                nn.LayerNorm(64+2),
                activation,
                add_noise_channel(),
                nn.TransformerEncoder(transformer_layer_3,n_layers),
                nn.LayerNorm(64+3),
                activation,
                transpose(),
                nn.ConvTranspose1d(64+3,self.n_dim+1,3),
                transpose(),
                nn.LayerNorm(self.n_dim+1),
                activation,
                nn.Linear(self.n_dim+1, self.n_dim),
                nn.LayerNorm(self.n_dim)
            )

    def get_delta(self, y_pixels, specified_distortion_budget=None, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere taken'''
        # Taken from viewmaker github

        distortion_budget = self.default_distortion_budget if specified_distortion_budget==None else specified_distortion_budget
        delta = torch.tanh(y_pixels) # Project to [-1, 1]

        avg_magnitude = delta.abs().mean([1,2], keepdim=True)

        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, specified_distortion_budget=None):
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_dim for i in range(x.shape[0])]

        seasonal, trend = self.decomp(x)

        out_s = self.netS(seasonal)
        out_t = self.netT(trend)

        combined = out_s+out_t
        delta = self.get_delta(combined, specified_distortion_budget)

        out = x+delta
        for i,length in enumerate(lengths):
            out[i,int(length):, ...] = -100

        return out

class DisruptMLPHead(nn.Module):
    def __init__(self, n_dim, activation, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_dim, hidden_dim),activation,nn.Linear(hidden_dim,hidden_dim),activation,nn.Linear(hidden_dim,1),nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class extract_tensor(nn.Module):
    def forward(self,x):
        return x[0]

class transpose(nn.Module):
    def forward(self,x):
        return x.transpose(1,2)

class add_noise_channel(nn.Module):
    def forward(self, x):
        return torch.cat((x,torch.rand(x[:,:,0].unsqueeze(-1).shape, device=x.device)),dim=-1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TestViewmaker(nn.Module):
    def __init__(self, n_dim, n_layers, layer_type, activation, default_distortion_budget, hidden_dim, n_head):
        super().__init__()
        self.n_dim=n_dim
        self.default_distortion_budget=default_distortion_budget
        self.vs = VNet(n_dim,n_layers,activation,hidden_dim,n_head)
        self.vt = VNet(n_dim,n_layers,activation,hidden_dim,n_head)
        self.decomp = series_decomp(25)
        self.add_noise = add_noise_channel()

    def get_delta(self, y_pixels, specified_distortion_budget=None, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere taken'''
        # Taken from viewmaker github
        delta=y_pixels
        distortion_budget =  self.default_distortion_budget if specified_distortion_budget==None else specified_distortion_budget
        avg_magnitude = delta.abs().mean([1,2], keepdim=True)

        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta

    def forward(self, x, specified_distortion_budget=None):
        lengths = [len(x[i])-len(torch.where(x[i]==-100)[0])/self.n_dim for i in range(x.shape[0])]

        seasonal, trend = self.decomp(x)
        seasonal = self.add_noise(seasonal)
        trend = self.add_noise(trend)

        out_s = self.vs(seasonal)
        out_t = self.vt(trend)

        out = out_s+out_t

        delta = self.get_delta(out,specified_distortion_budget)
        delta[:,:,5] = 0
        delta[:,:,8] = 0
        delta[:,:,11] = 0
        out = x+delta

        for i,length in enumerate(lengths):
            out[i,int(length):, ...] = -100

        return out

class VNet(nn.Module):
    def __init__(self, n_dim, n_layers, activation, hidden_dim, n_head):
        super().__init__()
        self.transpose = transpose()
        self.conv1 = nn.Conv1d(n_dim+1,64,3)
        self.norm1 = nn.LayerNorm(64)
        self.conv2 = nn.Conv1d(64,128,3)
        self.norm2 = nn.LayerNorm(128)
        self.deconv1 = nn.ConvTranspose1d(128,64,3)
        self.deconv2 = nn.ConvTranspose1d(64,n_dim+1,3)

        self.act = activation

        self.transformer_layer = nn.TransformerDecoderLayer(128,n_head,hidden_dim)
        self.transformer = nn.TransformerDecoder(self.transformer_layer,n_layers)

        self.out = nn.Linear(n_dim+1,n_dim)

    def forward(self, x):
        out = self.transpose(x)
        out = self.conv1(out)
        out = self.transpose(out)
        out = self.act(self.norm1(out))

        out = self.transpose(out)
        out = self.conv2(out)
        out = self.transpose(out)
        out = self.act(self.norm2(out))

        out = self.transformer(out,torch.rand(size=out.shape,device=out.device))
        out = self.act(self.norm2(out))

        out = self.transpose(out)
        out = self.deconv1(out)
        out = self.transpose(out)
        out = self.act(self.norm1(out))

        out = self.transpose(out)
        out = self.deconv2(out)
        out = self.transpose(out)
        out = self.out(out)

        return out