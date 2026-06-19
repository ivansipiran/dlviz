"""Animación didáctica de multi-head attention (Transformer).

En lugar de UNA atención, el Transformer corre varias "cabezas" en paralelo,
cada una con sus propias proyecciones Q, K, V. Lo interesante: cada cabeza puede
aprender un TIPO de relación distinto. Aquí se ilustra con cuatro cabezas con
especialidades interpretables sobre «el gato negro bebe leche fría»:

  * Sujeto–verbo (sintáctica)      : el verbo busca a su sujeto (bebe → gato)
  * Adjetivo–sustantivo            : cada adjetivo apunta a su sustantivo
  * Posición vecina (posicional)   : cada palabra mira a su vecina anterior
  * Verbo–objeto                   : el verbo se conecta con su objeto

Las salidas de todas las cabezas se CONCATENAN y se proyectan con W_O, fusionando
varias "vistas" de la frase en una representación más rica.

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from multi_head import multi_head_interactiva
    multi_head_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


TOKENS = ["el", "gato", "negro", "bebe", "leche", "fría"]
N = len(TOKENS)


def _head(links, self_b=0.5):
    s = np.full((N, N), -2.0)
    np.fill_diagonal(s, self_b)
    for i, j, w in links:
        s[i, j] = w
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


HEADS = [
    {"nombre": "Sujeto–verbo", "color": "#8e44ad", "cmap": "Purples",
     "desc": "Sintáctica: el verbo busca a su sujeto (bebe → gato).",
     "A": _head([(3, 1, 3.5), (1, 3, 2.2)])},
    {"nombre": "Adjetivo–sustantivo", "color": "#e67e22", "cmap": "Oranges",
     "desc": "Modificadores: cada adjetivo apunta a su sustantivo\n(negro → gato, fría → leche).",
     "A": _head([(2, 1, 3.5), (5, 4, 3.5)])},
    {"nombre": "Posición vecina", "color": "#16a085", "cmap": "Greens",
     "desc": "Posicional: cada palabra mira a su vecina anterior.",
     "A": _head([(i, i - 1, 3.2) for i in range(1, N)], self_b=0.4)},
    {"nombre": "Verbo–objeto", "color": "#2980b9", "cmap": "Blues",
     "desc": "Verbo–objeto: el verbo se conecta con su objeto (bebe → leche).",
     "A": _head([(3, 4, 3.5), (4, 3, 2.5)])},
]
H = len(HEADS)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _chip(ax, x, y, texto, face, fc="white", lw=1.4, w=1.05, h=0.56):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=face, edgecolor="#33425b", lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", color=fc, fontsize=10,
            fontweight="bold", zorder=4)


def _caja(ax, x, y, w, h, texto, fc, ec="#33425b", fs=8, tc="#222"):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 facecolor=fc, edgecolor=ec, lw=1.6, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def figura_multihead(head_sel=0):
    head_sel = int(np.clip(head_sel, 0, H - 1))
    HD = HEADS[head_sel]
    col = HD["color"]
    A = HD["A"]

    fig = plt.figure(figsize=(13.5, 8.4), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.32, 1.45, 0.62], width_ratios=[1, 1])
    ax_chips = fig.add_subplot(gs[0, :])
    ax_arcs = fig.add_subplot(gs[1, 1])
    ax_pipe = fig.add_subplot(gs[2, :])
    sub = gs[1, 0].subgridspec(2, 2, hspace=0.62, wspace=0.32)

    # ---- Frase ----
    ax_chips.set_xlim(-0.5, N * 1.2); ax_chips.set_ylim(-0.4, 0.8); ax_chips.axis("off")
    for k, tok in enumerate(TOKENS):
        _chip(ax_chips, k * 1.2, 0.2, tok, face="#eef2f7", fc="#222")
    ax_chips.set_title("Varias cabezas de atención en PARALELO, cada una "
                       "especializada en una relación distinta",
                       fontsize=11.5, fontweight="bold")

    # ---- Small multiples: las H matrices de atención ----
    ini = [t[:3] for t in TOKENS]
    for k, hd in enumerate(HEADS):
        axh = fig.add_subplot(sub[k // 2, k % 2])
        axh.imshow(hd["A"], cmap=hd["cmap"], vmin=0, vmax=1, aspect="equal")
        axh.set_xticks(range(N)); axh.set_xticklabels(ini, fontsize=6, rotation=45)
        axh.set_yticks(range(N)); axh.set_yticklabels(ini, fontsize=6)
        sel = (k == head_sel)
        for sp in axh.spines.values():
            sp.set_edgecolor(hd["color"]); sp.set_linewidth(3.0 if sel else 1.0)
            sp.set_visible(True)
        axh.set_title(f"cabeza {k}: {hd['nombre']}", fontsize=8.5,
                      fontweight="bold" if sel else "normal",
                      color=hd["color"] if sel else "#555")
    fig.text(0.045, 0.93, "Matriz de atención por cabeza (todas distintas):",
             fontsize=9.5, color="#444")

    # ---- Arcos: qué "ve" la cabeza seleccionada ----
    ax_arcs.set_xlim(-0.7, N * 1.3); ax_arcs.set_ylim(-1.4, 2.3); ax_arcs.axis("off")
    xs = np.arange(N) * 1.3
    for j, tok in enumerate(TOKENS):
        _chip(ax_arcs, xs[j], 0.0, tok, face="#eef2f7", fc="#222", w=1.1)
    for i in range(N):
        j = int(A[i].argmax()); w = A[i, j]
        if j == i:
            ax_arcs.add_patch(mpatches.Circle((xs[i], 0.62), 0.10, fill=False,
                              edgecolor=col, lw=1.0, alpha=0.35))
        else:
            rad = 0.5 if j > i else -0.5
            ax_arcs.add_patch(FancyArrowPatch((xs[i], 0.3), (xs[j], 0.3),
                              connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                              color=col, lw=1.2 + 5*w, alpha=0.85, zorder=2))
    ax_arcs.text(xs.mean(), -0.85, HD["desc"], ha="center", va="center",
                 fontsize=10, color=col, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col))
    ax_arcs.set_title(f"Cabeza {head_sel} en detalle: «{HD['nombre']}»\n"
                      "(flechas = a quién atiende cada palabra)",
                      fontsize=10.5, fontweight="bold", color=col)

    # ---- Pipeline: X -> cabezas -> concat -> W_O -> salida ----
    ax_pipe.set_xlim(0, 17); ax_pipe.set_ylim(-0.2, 3.0); ax_pipe.axis("off")
    _caja(ax_pipe, 1.0, 1.4, 1.5, 1.1, "X\n(tokens)", "#eef2f7", fs=9)
    ys = [2.5, 1.8, 1.1, 0.4]
    for k, hd in enumerate(HEADS):
        sel = (k == head_sel)
        ax_pipe.annotate("", xy=(3.0, ys[k]), xytext=(1.8, 1.4),
                         arrowprops=dict(arrowstyle="-|>", color="#bbb", lw=1.2))
        _caja(ax_pipe, 4.7, ys[k], 3.0, 0.52,
              f"cabeza {k}: $Q_{k}K_{k}V_{k}$→atención",
              "#fff" if not sel else hd["color"],
              ec=hd["color"], fs=7.5, tc=hd["color"] if not sel else "white")
        ax_pipe.annotate("", xy=(8.7, 1.45), xytext=(6.25, ys[k]),
                         arrowprops=dict(arrowstyle="-|>", color=hd["color"],
                                         lw=1.4, alpha=0.8))
    _caja(ax_pipe, 9.7, 1.45, 1.7, 1.4, "concat\n(unir\ncabezas)", "#f4ecf7", fs=8)
    ax_pipe.annotate("", xy=(11.4, 1.45), xytext=(10.55, 1.45),
                     arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))
    _caja(ax_pipe, 12.4, 1.45, 1.5, 1.0, "$W_O$\n(mezcla)", "#eef2f7", fs=9)
    ax_pipe.annotate("", xy=(14.1, 1.45), xytext=(13.15, 1.45),
                     arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=2))
    _caja(ax_pipe, 15.4, 1.45, 1.7, 1.2, "salida\nrica", "#e8f5e9", ec="#16a085", fs=9)
    ax_pipe.set_title("Las cabezas se calculan en paralelo y se fusionan: "
                      "concat + $W_O$", fontsize=10, fontweight="bold")

    fig.suptitle("Multi-head attention: varias miradas a la vez sobre la misma "
                 "frase", fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def multi_head_interactiva():
    """Despliega multi-head attention: explora qué relación captura cada cabeza.

    Controles: un selector de cabeza (resalta su matriz, dibuja sus arcos sobre
    la frase y describe su especialidad) y un botón para recorrer todas las
    cabezas una por una.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"h": 0}
    out = widgets.Output(layout=widgets.Layout(height="860px", overflow="hidden"))

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_multihead(estado["h"])
            plt.show()

    s_head = widgets.Dropdown(
        options=[(f"{k}: {hd['nombre']}", k) for k, hd in enumerate(HEADS)],
        value=0, description="cabeza", style={"description_width": "60px"},
        layout=widgets.Layout(width="290px"))
    b_run = widgets.Button(description="▶ Recorrer cabezas", button_style="success")

    def on_head(_):
        estado["h"] = s_head.value
        redibujar()
    s_head.observe(on_head, names="value")

    def on_run(_):
        for w in (b_run, s_head):
            w.disabled = True
        try:
            for k in range(H):
                estado["h"] = k
                s_head.unobserve(on_head, names="value")
                s_head.value = k
                s_head.observe(on_head, names="value")
                redibujar()
                time.sleep(1.6)
        finally:
            for w in (b_run, s_head):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Multi-head attention</h3>"
        "<span style='color:#555'>En vez de una sola atención, el Transformer usa "
        "<b>varias cabezas en paralelo</b>, y cada una puede aprender una relación "
        "distinta (sintáctica, posicional, verbo–objeto…). Luego se "
        "<b>concatenan</b> y se mezclan con W_O. Elige una cabeza para ver qué "
        "captura, o recórrelas todas.</span>")
    controles = widgets.HBox([s_head, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
