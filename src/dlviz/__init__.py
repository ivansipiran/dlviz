"""dlviz — Visualizaciones interactivas para Deep Learning."""

from .perceptron import perceptron_interactivo
from .mlp_xor import mlp_xor_interactivo

__all__ = ["perceptron_interactivo", "mlp_xor_interactivo"]
__version__ = "0.1.0"