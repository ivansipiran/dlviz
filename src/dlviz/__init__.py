"""dlviz — Visualizaciones interactivas para Deep Learning."""

from .perceptron import perceptron_interactivo
from .mlp_xor import mlp_xor_interactivo
from .optim import optimizadores_interactivo
from .atencion import atencion_interactiva
from .conv import conv_interactiva
from .pad_stride import pad_stride_interactiva
from .pooling import pooling_interactiva
from .multicanal import multicanal_interactiva
from .rnn import rnn_interactiva
from .lstm import lstm_interactiva
from .seq2seq import seq2seq_interactiva
from .nmt_atencion import nmt_atencion_interactiva
from .self_attention import self_attention_interactiva
from .multi_head import multi_head_interactiva
from .positional_encoding import positional_encoding_interactiva


__all__ = [
    "perceptron_interactivo",
    "mlp_xor_interactivo",
    "optimizadores_interactivo",
    "atencion_interactiva",
    "conv_interactiva",
    "pad_stride_interactiva",
    "pooling_interactiva",
    "multicanal_interactiva",
    "rnn_interactiva",
    "lstm_interactiva",
    "seq2seq_interactiva",
    "nmt_atencion_interactiva",
    "self_attention_interactiva",
    "multi_head_interactiva",
    "positional_encoding_interactiva"

]

__version__ = "0.2.0"