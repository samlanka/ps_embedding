#  Can later be changed to argparse
from dataclasses import dataclass


@dataclass
class TextEncoderConfig:
    embed_dim: int = 64
    snapshot_num_heads: int = 4
    snapshot_num_layers: int = 1
    choice_num_heads: int = 4
    choice_num_layers: int = 1
    feedforward_dim: int = 512


@dataclass
class TrainConfig:
    num_epochs: int = 1
    batch_size: int = 32

