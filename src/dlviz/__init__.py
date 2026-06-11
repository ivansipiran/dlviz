"""dlviz — Visualizaciones interactivas para Deep Learning."""

from .perceptron import perceptron_interactivo
from .mlp_xor import mlp_xor_interactivo
from .optim import optimizadores_interactivo
from .atencion import atencion_interactiva

__all__ = ["perceptron_interactivo", "mlp_xor_interactivo", "optimizadores_interactivo","atencion_interactiva"]
__version__ = "0.1.1"