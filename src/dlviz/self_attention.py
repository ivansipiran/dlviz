"""Animación didáctica de self-attention (la base del Transformer).

Conecta con la atención de la traducción, pero ahora la atención se aplica
DENTRO de una misma secuencia: cada token genera query, key y value a partir de
sí mismo, y atiende a todos los tokens (incluido él). A diferencia de una RNN,
no hay recurrencia: todas las posiciones se procesan EN PARALELO. Esa es la idea
central de «attention is all you need».

Se muestran: el pipeline X -> Q,K,V -> QKᵀ/√d -> softmax -> A -> A·V, la matriz
de atención NxN completa (cada token con todos), y, para el token elegido, a qué
atiende y cómo su salida es una mezcla ponderada.

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from self_attention import self_attention_interactiva
    self_attention_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


TOKENS = ["El", "gato", "bebe", "leche", "fría"]
# Embeddings semánticos: [DET, SUJETO, ACCIÓN, OBJETO, FRÍO]
_EMB = np.array([
    [1.0, 0.7, 0.0, 0.0, 0.0],
    [0.3, 1.0, 0.5, 0.0, 0.0],
    [0.0, 0.6, 1.0, 0.6, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.5],
    [0.0, 0.0, 0.0, 0.6, 1.0],
])
_EMB = _EMB / np.linalg.norm(_EMB, axis=1, keepdims=True)
N, Dd = _EMB.shape
COLS = ["#8e44ad", "#e67e22", "#16a085", "#2980b9", "#c0392b"]


def atencion(nitidez=4.0):
    """Matriz de self-attention NxN (cada fila suma 1)."""
    s = (_EMB @ _EMB.T) * nitidez
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _caja(ax, x, y, w, h, texto, fc="white", ec="#33425b", fs=9, tc="#222"):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fc, edgecolor=ec, lw=1.6, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def _flecha(ax, x0, x1, y, texto="", color="#7f8c8d"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    if texto:
        ax.text((x0 + x1) / 2, y + 0.28, texto, ha="center", fontsize=8, color="#555")


def _chip(ax, x, y, texto, face, fc="white", lw=1.6, w=1.2, h=0.6):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=face, edgecolor="#33425b", lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", color=fc, fontsize=11,
            fontweight="bold", zorder=4)


def figura_self_attention(q=2, nitidez=4.0):
    A = atencion(nitidez)
    q = int(np.clip(q, 0, N - 1))
    col = COLS[q]

    fig = plt.figure(figsize=(13, 7.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.7, 1.3], width_ratios=[1, 1])
    ax_pipe = fig.add_subplot(gs[0, :])
    ax_mat = fig.add_subplot(gs[1, 0])
    ax_flow = fig.add_subplot(gs[1, 1])

    # ---- Pipeline X -> Q,K,V -> scores -> softmax -> A -> A·V ----
    ax_pipe.set_xlim(0, 16); ax_pipe.set_ylim(-0.4, 2.4); ax_pipe.axis("off")
    _caja(ax_pipe, 1.1, 1.3, 1.6, 1.2, "X\n(tokens)\n%d×d" % N, fc="#eef2f7")
    _flecha(ax_pipe, 1.9, 3.1, 1.3, "$W_Q,W_K,W_V$")
    for k, (nm, yy) in enumerate(zip(["Q", "K", "V"], [1.9, 1.3, 0.7])):
        _caja(ax_pipe, 3.9, yy, 1.0, 0.46, nm, fc="#fdf3e6", fs=10)
    _flecha(ax_pipe, 4.5, 6.0, 1.3)
    _caja(ax_pipe, 7.0, 1.3, 1.9, 1.0, "$QK^\\top/\\sqrt{d}$\n(scores)\n%d×%d" % (N, N),
          fc="#eef2f7")
    _flecha(ax_pipe, 8.0, 9.4, 1.3, "softmax")
    _caja(ax_pipe, 10.4, 1.3, 1.7, 1.0, "A\n(pesos)\n%d×%d" % (N, N), fc="#fdeaea",
          ec="#c0392b")
    _flecha(ax_pipe, 11.3, 12.7, 1.3, "× V")
    _caja(ax_pipe, 14.0, 1.3, 1.9, 1.2,
          "salida\ncada token\ncontextualizado", fc="#e8f5e9", ec="#16a085")
    ax_pipe.text(8, -0.2, "Self-attention: Q, K y V salen de la MISMA secuencia · "
                 "todas las posiciones se calculan EN PARALELO (sin recurrencia) → "
                 "base del Transformer", ha="center", fontsize=9.5, color="#16a085",
                 fontweight="bold")
    ax_pipe.set_title("Mecanismo de self-attention", fontsize=12, fontweight="bold")

    # ---- Matriz de atención NxN ----
    ax_mat.imshow(A, cmap="Purples", vmin=0, vmax=1, aspect="equal")
    for i in range(N):
        for j in range(N):
            ax_mat.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if A[i, j] > 0.55 else "#222")
    ax_mat.set_xticks(range(N)); ax_mat.set_xticklabels(TOKENS, fontsize=8, rotation=25)
    ax_mat.set_yticks(range(N)); ax_mat.set_yticklabels(TOKENS, fontsize=8)
    ax_mat.set_xlabel("key (a quién mira)", fontsize=9)
    ax_mat.set_ylabel("query (quién mira)", fontsize=9)
    ax_mat.add_patch(mpatches.Rectangle((-0.5, q - 0.5), N, 1, fill=False,
                     edgecolor=col, lw=3, zorder=5))
    ax_mat.set_title("Matriz de atención: cada token con todos\n(cada fila suma 1)",
                     fontsize=10.5, fontweight="bold")

    # ---- Flujo de self-attention para el token query ----
    ax_flow.set_xlim(-0.6, N * 1.45); ax_flow.set_ylim(-1.3, 2.3); ax_flow.axis("off")
    xs = np.arange(N) * 1.45
    for j, tok in enumerate(TOKENS):
        _chip(ax_flow, xs[j], 0.0, tok,
              face=col if j == q else "#dfe3e8",
              fc="white" if j == q else "#555",
              lw=3 if j == q else 1.2)
    # arcos de atención desde el query hacia todos los tokens
    for j in range(N):
        w = A[q, j]
        if j == q:
            ax_flow.add_patch(mpatches.Circle((xs[q], 0.95), 0.16 + 0.5*w,
                              fill=False, edgecolor=col, lw=0.8 + 5*w, alpha=0.5))
        else:
            rad = 0.45 if j > q else -0.45
            arc = FancyArrowPatch((xs[q], 0.32), (xs[j], 0.32),
                                  connectionstyle=f"arc3,rad={rad}",
                                  arrowstyle="-|>", color=col, lw=0.8 + 6*w,
                                  alpha=0.25 + 0.7*w, zorder=2)
            ax_flow.add_patch(arc)
        ax_flow.text(xs[j], -0.7, f"{w:.2f}", ha="center", fontsize=8,
                     color=col if w > 0.15 else "#999",
                     fontweight="bold" if j == A[q].argmax() else "normal")
    ax_flow.text(xs.mean(), -1.15, f"salida(«{TOKENS[q]}») = Σ pesos · value  "
                 "(mezcla ponderada de toda la secuencia)", ha="center",
                 fontsize=9, color="#333")
    ax_flow.set_title(f"«{TOKENS[q]}» atiende a toda la secuencia "
                      f"(sobre todo a «{TOKENS[A[q].argmax()]}»)",
                      fontsize=10.5, fontweight="bold")

    fig.suptitle("Self-attention: cada palabra mira a todas las demás, en paralelo",
                 fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def self_attention_interactiva():
    """Despliega la self-attention interactiva sobre una frase de ejemplo.

    Controles: el token que consulta (query), la nitidez del softmax y un botón
    para recorrer todos los tokens (recordando que en realidad se calculan a la
    vez, en paralelo).
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"q": 2}
    # Altura fija + overflow hidden: evita que aparezca/desaparezca la barra de
    # scroll de Colab al pasar el mouse (lo que provoca el temblor por re-layout).
    out = widgets.Output(layout=widgets.Layout(height="740px", overflow="hidden"))

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_self_attention(estado["q"], s_nit.value)
            plt.show()

    s_q = widgets.Dropdown(options=[(t, i) for i, t in enumerate(TOKENS)], value=2,
                           description="query", style={"description_width": "60px"},
                           layout=widgets.Layout(width="230px"))
    s_nit = widgets.FloatSlider(value=4.0, min=0.5, max=9.0, step=0.5,
                                description="nitidez (escala softmax)",
                                style={"description_width": "160px"},
                                layout=widgets.Layout(width="370px"),
                                readout_format=".1f")
    b_run = widgets.Button(description="▶ Recorrer tokens", button_style="success")

    def on_q(_):
        estado["q"] = s_q.value
        redibujar()
    s_q.observe(on_q, names="value")
    s_nit.observe(lambda _: redibujar(), names="value")

    def on_run(_):
        for w in (b_run, s_q, s_nit):
            w.disabled = True
        try:
            for i in range(N):
                estado["q"] = i
                s_q.unobserve(on_q, names="value")
                s_q.value = i
                s_q.observe(on_q, names="value")
                redibujar()
                time.sleep(0.9)
        finally:
            for w in (b_run, s_q, s_nit):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Self-attention (Transformer)</h3>"
        "<span style='color:#555'>Igual que la atención de la traducción, pero "
        "<b>dentro de una sola secuencia</b>: cada token genera Q, K, V de sí "
        "mismo y atiende a todos. No hay recurrencia: todo se calcula <b>en "
        "paralelo</b>. Elige el token que consulta y mira a qué atiende.</span>")
    controles = widgets.VBox([widgets.HBox([s_q, b_run]), s_nit])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
