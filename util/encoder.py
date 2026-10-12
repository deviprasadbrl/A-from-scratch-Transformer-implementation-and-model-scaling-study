import torch
import torch.nn as nn
import math
from util.util import MultiHeadAttention,PositionalEncoding
from config import Config

class Encoder_layers(nn.Module):
    def __init__(self,d_model,num_heads,ff_dim,dropout_rate):
        super().__init__()

        self.attn=MultiHeadAttention(d_model,num_heads)

        self.ff=nn.Sequential(
            nn.Linear(d_model,ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim,d_model)
        )

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x,mask=None):
        normed_x1=self.norm1(x)
        attn=self.attn(normed_x1,mask=mask,is_causal=False)
        x=x+self.dropout(attn)

        normed_x2=self.norm2(x)
        ff_out=self.ff(normed_x2)
        x=x+self.dropout(ff_out)

        return x

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,num_blocks,vocab_size,max_length,ff_dim,dropout_rate):
        super().__init__()
        self.d_model=d_model

        self.embed=nn.Embedding(vocab_size,d_model)

        self.pos=PositionalEncoding(d_model,max_length)

        self.layers=nn.ModuleList([Encoder_layers(d_model,num_heads,ff_dim,dropout_rate) for _ in range(num_blocks)])

        self.norm=nn.LayerNorm(d_model)

    def forward(self,x,mask=None):
        x=self.embed(x)*math.sqrt(self.d_model)
        x=self.pos(x)

        for layers in self.layers:
            x=layers(x,mask=mask)

        x=self.norm(x)

        return x

