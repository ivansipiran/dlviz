"""dlviz — Visualizaciones interactivas para Deep Learning."""

from .perceptron import perceptron_interactivo
from .mlp_xor import mlp_xor_interactivo
from .optim import optimizadores_interactivo
from .atencion import atencion_interactiva
from .conv import conv_interactiva
from .pad_stride import pad_stride_interactiva
from .pooling import pooling_interactiva
from .multicanal import multicanal_interactiva

__all__ = ["perceptron_interactivo", "mlp_xor_interactivo", "optimizadores_interactivo","atencion_interactiva","conv_interactiva","pad_stride_interactiva","pooling_interactiva","multicanal_interactiva"]
__version__ = "0.1.5"