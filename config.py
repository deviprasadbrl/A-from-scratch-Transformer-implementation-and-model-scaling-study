from dataclasses import dataclass
from dataset_processing.shared_tokenizer import VOCAB_SIZE


@dataclass
class Config:
    vocab_size: int=VOCAB_SIZE
    d_model: int=512
    num_heads: int=8
    encoder_blocks:int=6
    decoder_blocks: int=6
    ff_dim: int=2048
    max_length: int=160
    dropout_rate: float=0.1

    lr: float=3e-4
    total_epochs: int=40
    max_norm: float=1.0
    batch_size: int=64
    gradient_accumulation: int=2
    weight_decay: float=0.01


# 23.36M parameter Transformer
CONFIG_23M=Config(
    d_model=256,
    ff_dim=1024,
    batch_size=128,
    total_epochs=30,
)

# 68.73M parameter Transformer
CONFIG_68M=Config(
    d_model=512,
    ff_dim=2048,
    batch_size=64,
    total_epochs=15,
)