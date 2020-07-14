import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from remar.common.util import get_encoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, 
                embed: nn.Embedding = None,
                ntoken: int = 150000, 
                hidden_size:  int = 200,
                # ninp: int = 2048, 
                nhead: int = 2, 
                nhid: int = 200, 
                nlayers:int = 2, 
                dropout: float = 0.1,
                layer:        str = "lstm",):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(
            embed,
            nn.Dropout(p=dropout)
        )

        self.enc_layer = get_encoder(layer, emb_size, hidden_size)
        
        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.pos_encoder = PositionalEncoding(enc_size, dropout)

        encoder_layers = TransformerEncoderLayer(enc_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = enc_size
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, z):

        rnn_mask = mask
        emb = self.embed_layer(x)

        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _ , y = self.enc_layer(emb, rnn_mask, lengths)
        y_extended = y.unsqueeze(0)
        
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     # device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src))
        #     # .to(device)
        #     self.src_mask = mask

        # # Not use encoder and position encoder this time
        # src = src.long()
        # y_extended = y_extended * math.sqrt(self.ninp)
        # y_extended = self.pos_encoder(y_extended)
        output = self.transformer_encoder(y_extended)
        # print(output.shape)
        # output = self.decoder(output)
        return output, y