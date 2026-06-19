"""Animación didáctica de una red neuronal recurrente (RNN) simple.

Muestra cómo una RNN procesa una secuencia de palabras paso a paso, manteniendo
un estado oculto (la "memoria") que se pasa hacia adelante. En cada paso aplica
la misma recurrencia con los mismos pesos:

    h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from rnn import rnn_interactiva
    rnn_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Secuencia, embeddings y pesos (fijos, para un demo reproducible)
# ---------------------------------------------------------------------------
TOKENS = ["no", "me", "gustó", "la", "película"]
T = len(TOKENS)
D = 4          # dimensión del embedding de entrada
H = 5          # dimensión del estado oculto

_EMB = np.random.default_rng(3).normal(0, 1, (T, D))
_rng = np.random.default_rng(7)
_WXH = _rng.normal(0, 0.6, (H, D))
_WHH = _rng.normal(0, 0.5, (H, H))
_B = _rng.normal(0, 0.1, H)


def estados():
    """Devuelve los estados ocultos h_0..h_T como un arreglo (T+1, H). h_0 = 0."""
    h = np.zeros(H)
    todos = [h.copy()]
    for t in range(T):
        h = np.tanh(_WXH @ _EMB[t] + _WHH @ h + _B)
        todos.append(h.copy())
    return np.array(todos)


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------
def _fila(ax, x0, y0, vals, etiqueta, cmap="coolwarm", vmin=-1, vmax=1, cw=0.82):
    cm = plt.get_cmap(cmap)
    ax.text(x0 - 0.25, y0 + 0.4, etiqueta, ha="right", va="center", fontsize=9.5)
    for j, v in enumerate(vals):
        t = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
        rgb = cm(t)[:3]
        ax.add_patch(mpatches.Rectangle((x0 + j * cw, y0), cw, 0.8, facecolor=rgb,
                     edgecolor="#888", lw=0.7))
        tc = "white" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]) < 0.5 else "#111"
        ax.text(x0 + j * cw + cw / 2, y0 + 0.4, f"{v:+.2f}", ha="center",
                va="center", fontsize=7.5, color=tc)


def _chip(ax, x, y, texto, face, edge="#33425b", lw=1.4, w=1.15, h=0.62, fc="white"):
    ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=face, edgecolor=edge, lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", color=fc, fontsize=11,
            fontweight="bold", zorder=4)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
def figura_rnn(paso=0):
    S = estados()                  # (T+1, H)
    paso = int(np.clip(paso, 0, T - 1))
    x_t = _EMB[paso]
    h_prev = S[paso]
    h_new = S[paso + 1]

    fig = plt.figure(figsize=(13, 8.6), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1.0, 1.5], width_ratios=[1, 1])
    ax_chips = fig.add_subplot(gs[0, :])
    ax_chain = fig.add_subplot(gs[1, :])
    ax_cell = fig.add_subplot(gs[2, 0])
    ax_heat = fig.add_subplot(gs[2, 1])

    # --- Secuencia de palabras (cabeza de lectura) ---
    ax_chips.set_xlim(-0.5, T * 1.6)
    ax_chips.set_ylim(-0.5, 1.4)
    ax_chips.axis("off")
    for i, tok in enumerate(TOKENS):
        x = i * 1.6
        leido = i <= paso
        _chip(ax_chips, x, 0.2, tok,
              face="#e67e22" if i == paso else ("#5b6b80" if leido else "#dfe3e8"),
              fc="white" if (i == paso or leido) else "#777",
              lw=3 if i == paso else 1.2)
        if i == paso:
            ax_chips.plot([x], [0.95], marker="v", ms=14, color="#e67e22")
    ax_chips.set_title(f"Secuencia procesada en orden  ·  paso t = {paso+1} de {T}  "
                       f"·  leyendo «{TOKENS[paso]}»", fontsize=12, fontweight="bold")

    # --- RNN desenrollada en el tiempo ---
    ax_chain.set_xlim(-0.9, T + 0.7)
    ax_chain.set_ylim(-1.8, 1.6)
    ax_chain.axis("off")
    ax_chain.add_patch(mpatches.Circle((0, 0), 0.16, facecolor="#eee",
                       edgecolor="#888", zorder=3))
    ax_chain.text(0, 0, "$h_0$", ha="center", va="center", fontsize=9, zorder=4)
    for t in range(1, T + 1):
        activa = (t == paso + 1)
        pasada = (t <= paso + 1)
        col = "#e67e22" if activa else ("#5b6b80" if pasada else "#cfd5dc")
        # flecha de estado h_{t-1} -> celda
        ax_chain.annotate("", xy=(t - 0.32, 0), xytext=(t - 1 + 0.18, 0),
                          arrowprops=dict(arrowstyle="-|>", lw=2,
                                          color=col if pasada else "#cfd5dc"))
        # celda recurrente
        ax_chain.add_patch(mpatches.FancyBboxPatch((t - 0.32, -0.34), 0.64, 0.68,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor="#fdecea" if activa else "white",
                           edgecolor=col, lw=3 if activa else 1.6, zorder=3))
        ax_chain.text(t, 0, "RNN", ha="center", va="center", fontsize=8.5,
                      fontweight="bold", color=col, zorder=4)
        # entrada x_t desde abajo
        ax_chain.annotate("", xy=(t, -0.36), xytext=(t, -1.15),
                          arrowprops=dict(arrowstyle="-|>", lw=1.8,
                                          color=col if pasada else "#cfd5dc"))
        ax_chain.text(t, -1.45, TOKENS[t - 1], ha="center", va="center",
                      fontsize=9, color="#333" if pasada else "#aaa")
        ax_chain.text(t - 0.5, 0.22, f"$h_{{{t-1}}}$", ha="center", fontsize=8,
                      color="#666")
    ax_chain.annotate("", xy=(T + 0.55, 0), xytext=(T + 0.18, 0),
                      arrowprops=dict(arrowstyle="-|>", lw=2, color="#5b6b80"))
    ax_chain.text(T + 0.62, 0.22, f"$h_{{{T}}}$", fontsize=8, color="#666")
    ax_chain.text(-0.8, -1.45, "entradas:", ha="left", fontsize=9, color="#555")
    ax_chain.set_title("RNN desenrollada: la misma celda (mismos pesos "
                       r"$W_{xh}, W_{hh}, b$) se reutiliza en cada paso; el estado "
                       r"$h$ se pasa hacia adelante", fontsize=10.5, fontweight="bold")

    # --- Cómputo de la celda en el paso actual ---
    ax_cell.set_xlim(0, 6.5)
    ax_cell.set_ylim(0, 6.2)
    ax_cell.axis("off")
    ax_cell.set_title(f"Cómputo en el paso t = {paso+1}", fontsize=11,
                      fontweight="bold")
    _fila(ax_cell, 2.0, 5.2, x_t, r"$x_t$  («%s»)" % TOKENS[paso], cmap="PuOr")
    _fila(ax_cell, 2.0, 4.1, h_prev, r"$h_{t-1}$  (estado previo)")
    ax_cell.text(0.1, 2.9, r"$h_t=\tanh\,(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b)$",
                 fontsize=12, va="center")
    ax_cell.text(0.1, 2.25, "Los mismos pesos se usan en todos los pasos "
                 "(compartición en el tiempo).", fontsize=8.5, color="#666",
                 va="center")
    _fila(ax_cell, 2.0, 1.0, h_new, r"$h_t$  (nuevo estado)")
    ax_cell.add_patch(mpatches.Rectangle((2.0 - 0.02, 1.0 - 0.04),
                      H * 0.82 + 0.04, 0.88, fill=False, edgecolor="#c0392b",
                      lw=2.4, zorder=5))

    # --- Estado oculto a lo largo del tiempo ---
    M = S[1:].T.copy()             # (H, T)
    mask = np.zeros_like(M, dtype=bool)
    mask[:, paso + 1:] = True
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f0f0f0")
    ax_heat.imshow(np.ma.array(M, mask=mask), cmap=cmap, vmin=-1, vmax=1,
                   aspect="auto", origin="upper")
    for i in range(H):
        for j in range(paso + 1):
            ax_heat.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                         fontsize=7.5,
                         color="white" if abs(M[i, j]) > 0.6 else "#222")
    ax_heat.set_xticks(range(T)); ax_heat.set_xticklabels(TOKENS, fontsize=8, rotation=20)
    ax_heat.set_yticks(range(H)); ax_heat.set_yticklabels([f"$h^{{({i})}}$" for i in range(H)],
                                                          fontsize=8)
    ax_heat.add_patch(mpatches.Rectangle((paso - 0.5, -0.5), 1, H, fill=False,
                      edgecolor="#c0392b", lw=2.5, zorder=5))
    ax_heat.set_title("Estado oculto en el tiempo (la 'memoria' que se actualiza)",
                      fontsize=10.5, fontweight="bold")

    fig.suptitle("Red neuronal recurrente: procesa la secuencia paso a paso y "
                 "arrastra un estado oculto", fontsize=12, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def rnn_interactiva():
    """Despliega la RNN interactiva: procesa la frase palabra por palabra.

    Controles: un slider de paso (qué palabra se está procesando) y un botón para
    animar el recorrido de toda la secuencia. En cada paso se ve la celda
    recurrente operando y cómo se actualiza el estado oculto.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_rnn(estado["paso"])
            plt.show()

    s_paso = widgets.IntSlider(value=0, min=0, max=T - 1, step=1,
                               description="paso t",
                               style={"description_width": "60px"},
                               layout=widgets.Layout(width="430px"),
                               continuous_update=False)
    b_run = widgets.Button(description="▶ Procesar secuencia", button_style="success")

    def on_paso(_):
        estado["paso"] = s_paso.value
        redibujar()
    s_paso.observe(on_paso, names="value")

    def on_run(_):
        for w in (b_run, s_paso):
            w.disabled = True
        try:
            for t in range(T):
                estado["paso"] = t
                redibujar()
                time.sleep(1.0)
            s_paso.unobserve(on_paso, names="value")
            s_paso.value = T - 1
            s_paso.observe(on_paso, names="value")
        finally:
            for w in (b_run, s_paso):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Red neuronal recurrente (RNN)</h3>"
        "<span style='color:#555'>La RNN lee la frase <b>palabra por palabra</b>. "
        "En cada paso combina la palabra actual con el <b>estado oculto</b> "
        "anterior (su memoria) usando <b>los mismos pesos</b>, y produce un nuevo "
        "estado. Avanza con el slider o pulsa Procesar.</span>")
    controles = widgets.HBox([s_paso, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
