# dlviz

Visualizaciones interactivas para el curso de Deep Learning (DCC, U. de Chile).

## Instalación
```bash
pip install dlviz
```

## Uso
```python
from dlviz import backprop_interactivo

backprop_interactivo()   # backpropagation paso a paso en un MLP 2-2-2
```

## Demos disponibles

| Tema | Función | Qué muestra |
|---|---|---|
| Perceptrón | `perceptron_interactivo()` | Forward de un perceptrón y su frontera de decisión |
| MLP | `mlp_xor_interactivo()` | MLP 2-H-1 aprendiendo XOR |
| Backpropagation | `backprop_interactivo()` | Forward, error y costo, δ y gradientes, actualización de pesos, paso a paso |
| Optimización | `optimizadores_interactivo()` | SGD, Momentum, RMSProp y Adam en superficies de pérdida |
| Convolución | `conv_interactiva()` | Ventana deslizante y mapa de salida |
| Padding y stride | `pad_stride_interactiva()` | Efecto de padding y stride en el tamaño de salida |
| Pooling | `pooling_interactiva()` | Max y average pooling |
| Multicanal | `multicanal_interactiva()` | Filtros sobre imágenes RGB y volumen de salida |
| RNN | `rnn_interactiva()` | RNN desenrollada y estado oculto en el tiempo |
| LSTM | `lstm_interactiva()` | Compuertas y estado de celda |
| Seq2seq | `seq2seq_interactiva()` | Encoder-decoder para traducción |
| Atención en NMT | `nmt_atencion_interactiva()` | Distribución de atención y matriz de alineación |
| Atención | `atencion_interactiva()` | Primer contacto con queries, keys y values |
| Self-attention | `self_attention_interactiva()` | Q, K, V y matriz de atención NxN |
| Positional encoding | `positional_encoding_interactiva()` | Codificación sinusoidal de posiciones |
| Multi-head attention | `multi_head_interactiva()` | Varias cabezas especializadas y concat + W_O |
