import torch
import math
import torch.nn as nn
from util.util import MultiHeadAttention,PositionalEncoding

class Decoder_Layers(nn.Module):
    def __init__(self,d_model,num_heads,ff_dim,dropout_rate):
        super().__init__()

        self.attn=MultiHeadAttention(d_model=d_model,num_heads=num_heads)

        self.cross_attn=MultiHeadAttention(d_model=d_model,num_heads=num_heads)

        self.ff_out=nn.Sequential(
            nn.Linear(d_model,ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim,d_model)
        )

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x,context,self_mask=None,cross_mask=None):
        norm_x1=self.norm1(x)
        attention=self.attn(norm_x1,mask=self_mask,is_causal=True)
        x=x+self.dropout(attention)

        norm_x2=self.norm2(x)
        cross_attnention=self.cross_attn(norm_x2,mask=cross_mask,is_causal=False,context=context)
        x=x+self.dropout(cross_attnention)

        norm_x3=self.norm3(x)
        ff_out=self.ff_out(norm_x3)
        x=x+self.dropout(ff_out)

        return x

class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,num_blocks,vocab_size,max_length,ff_dim,dropout_rate):
        super().__init__()

        self.d_model=d_model

        self.embed=nn.Embedding(vocab_size,d_model)

        self.pos=PositionalEncoding(d_model=d_model,max_length=max_length)

        self.layers=nn.ModuleList([Decoder_Layers(d_model,num_heads,ff_dim,dropout_rate) for _ in range(num_blocks)])

        self.norm=nn.LayerNorm(d_model)
        self.final_proj=nn.Linear(d_model,vocab_size)

    def forward(self,x,context,self_mask=None, cross_mask=None):
        x=self.embed(x)*math.sqrt(self.d_model)

        x=self.pos(x)

        for layers in self.layers:
            x=layers(x,context,self_mask=self_mask, cross_mask=cross_mask)
        
        x=self.norm(x)

        x=self.final_proj(x)

        return x


