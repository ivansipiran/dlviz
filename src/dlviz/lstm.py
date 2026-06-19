"""Animación didáctica de una celda LSTM.

Muestra cómo una LSTM procesa una secuencia manteniendo dos estados: el estado
de celda C (la "cinta transportadora" de memoria a largo plazo) y el estado
oculto h (la salida). Tres compuertas regulan la memoria en cada paso:

    f_t = σ(W_f·[h_{t-1},x_t] + b_f)     (forget : qué borrar de C)
    i_t = σ(W_i·[h_{t-1},x_t] + b_i)     (input  : cuánto escribir)
    g_t = tanh(W_g·[h_{t-1},x_t] + b_g)  (candidato: qué escribir)
    o_t = σ(W_o·[h_{t-1},x_t] + b_o)     (output : qué exponer)
    C_t = f_t · C_{t-1} + i_t · g_t      (· es elemento a elemento)
    h_t = o_t · tanh(C_t)

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from lstm import lstm_interactiva
    lstm_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


TOKENS = ["no", "me", "gustó", "la", "película"]
T = len(TOKENS)
D = 4          # dimensión del embedding
H = 4          # dimensión de los estados C y h

_EMB = np.random.default_rng(3).normal(0, 1, (T, D))
_rng = np.random.default_rng(11)
_Wf = _rng.normal(0, 0.5, (H, H + D))
_Wi = _rng.normal(0, 0.5, (H, H + D))
_Wg = _rng.normal(0, 0.5, (H, H + D))
_Wo = _rng.normal(0, 0.5, (H, H + D))
_bf = np.full(H, 1.0)          # sesgo de olvido positivo: recordar por defecto
_bi = _rng.normal(0, 0.2, H)
_bg = _rng.normal(0, 0.2, H)
_bo = _rng.normal(0, 0.2, H)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_estados():
    """Corre la LSTM y devuelve un dict con compuertas y estados por paso."""
    h = np.zeros(H); C = np.zeros(H)
    Cs = [C.copy()]; hs = [h.copy()]
    fs, iss, gs, os = [], [], [], []
    for t in range(T):
        z = np.concatenate([h, _EMB[t]])
        f = _sig(_Wf @ z + _bf)
        i = _sig(_Wi @ z + _bi)
        g = np.tanh(_Wg @ z + _bg)
        o = _sig(_Wo @ z + _bo)
        C = f * C + i * g
        h = o * np.tanh(C)
        fs.append(f); iss.append(i); gs.append(g); os.append(o)
        Cs.append(C.copy()); hs.append(h.copy())
    return {"C": np.array(Cs), "h": np.array(hs), "f": np.array(fs),
            "i": np.array(iss), "g": np.array(gs), "o": np.array(os)}


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _luz(rgb):
    return "white" if (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) < 0.5 else "#111"


def _barra(ax, x0, y0, vals, cmap, vmin, vmax, etiqueta="", desc="", cw=0.7):
    cm = plt.get_cmap(cmap)
    for j, v in enumerate(vals):
        t = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
        rgb = cm(t)[:3]
        ax.add_patch(mpatches.Rectangle((x0 + j*cw, y0), cw, 0.7, facecolor=rgb,
                     edgecolor="#888", lw=0.7))
        ax.text(x0 + j*cw + cw/2, y0 + 0.35, f"{v:+.1f}", ha="center", va="center",
                fontsize=7, color=_luz(rgb))
    if etiqueta:
        ax.text(x0 + len(vals)*cw/2, y0 + 1.02, etiqueta, ha="center", fontsize=9,
                fontweight="bold")
    if desc:
        ax.text(x0 + len(vals)*cw/2, y0 - 0.32, desc, ha="center", fontsize=7.5,
                color="#666")
    return x0 + len(vals)*cw


def _sim(ax, x, y, s, fs=15):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs)


def _chip(ax, x, y, texto, face, fc="white", lw=1.4, w=1.15, h=0.62):
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=face, edgecolor="#33425b", lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", color=fc, fontsize=11,
            fontweight="bold", zorder=4)


def figura_lstm(paso=0):
    S = lstm_estados()
    paso = int(np.clip(paso, 0, T - 1))
    C_prev, C_t = S["C"][paso], S["C"][paso + 1]
    h_t = S["h"][paso + 1]
    f, i, g, o = S["f"][paso], S["i"][paso], S["g"][paso], S["o"][paso]
    tanhC = np.tanh(C_t)

    GATE = "YlGn"      # compuertas sigmoide en [0,1] (válvulas)
    VAL = "coolwarm"   # valores en [-1,1]

    fig = plt.figure(figsize=(13, 9.6), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1.5, 1.0], width_ratios=[1, 1])
    ax_chips = fig.add_subplot(gs[0, :])
    ax_cell = fig.add_subplot(gs[1, :])
    ax_C = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])

    # --- Secuencia ---
    ax_chips.set_xlim(-0.5, T * 1.6); ax_chips.set_ylim(-0.5, 1.4); ax_chips.axis("off")
    for k, tok in enumerate(TOKENS):
        x = k * 1.6
        _chip(ax_chips, x, 0.2, tok,
              face="#e67e22" if k == paso else ("#5b6b80" if k < paso else "#dfe3e8"),
              fc="white" if k <= paso else "#777", lw=3 if k == paso else 1.2)
        if k == paso:
            ax_chips.plot([x], [0.95], marker="v", ms=14, color="#e67e22")
    ax_chips.set_title(f"Secuencia paso a paso  ·  t = {paso+1}/{T}  ·  "
                       f"leyendo «{TOKENS[paso]}»", fontsize=12, fontweight="bold")

    # --- Celda LSTM (ecuación desplegada como barras) ---
    ax_cell.set_xlim(0, 19.5); ax_cell.set_ylim(-1.2, 6.2)
    ax_cell.set_aspect("equal"); ax_cell.axis("off")
    ax_cell.text(0.2, 5.8, "Compuertas σ ∈ [0,1] (válvulas) calculadas desde "
                 r"$[h_{t-1},\,x_t]$.   « × » es elemento a elemento.",
                 fontsize=9.5, color="#444")

    # Fila A: actualización de la memoria  C_t = f·C_{t-1} + i·g
    yA = 3.6
    x = _barra(ax_cell, 0.3, yA, C_prev, VAL, -1, 1, "$C_{t-1}$", "memoria previa")
    _sim(ax_cell, x + 0.45, yA + 0.35, "×")
    x = _barra(ax_cell, x + 0.9, yA, f, GATE, 0, 1, "f", "olvidar (0=borra)")
    _sim(ax_cell, x + 0.45, yA + 0.35, "+")
    x = _barra(ax_cell, x + 0.9, yA, i, GATE, 0, 1, "i", "cuánto escribir")
    _sim(ax_cell, x + 0.45, yA + 0.35, "×")
    x = _barra(ax_cell, x + 0.9, yA, g, VAL, -1, 1, "g", "candidato")
    _sim(ax_cell, x + 0.45, yA + 0.35, "=")
    x = _barra(ax_cell, x + 0.9, yA, C_t, VAL, -1, 1, "$C_t$", "memoria nueva")
    ax_cell.add_patch(mpatches.Rectangle((x - H*0.7 - 0.04, yA - 0.04),
                      H*0.7 + 0.08, 0.78, fill=False, edgecolor="#16a085",
                      lw=2.4, zorder=5))

    # Fila B: salida  h_t = o · tanh(C_t)
    yB = 1.0
    x = _barra(ax_cell, 0.3, yB, tanhC, VAL, -1, 1, "$\\tanh(C_t)$", "")
    _sim(ax_cell, x + 0.45, yB + 0.35, "×")
    x = _barra(ax_cell, x + 0.9, yB, o, GATE, 0, 1, "o", "exponer")
    _sim(ax_cell, x + 0.45, yB + 0.35, "=")
    x = _barra(ax_cell, x + 0.9, yB, h_t, VAL, -1, 1, "$h_t$", "salida / a la sgte. celda")
    ax_cell.add_patch(mpatches.Rectangle((x - H*0.7 - 0.04, yB - 0.04),
                      H*0.7 + 0.08, 0.78, fill=False, edgecolor="#c0392b",
                      lw=2.4, zorder=5))

    ax_cell.annotate("", xy=(0.6, yB + 1.0), xytext=(0.6, yA - 0.1),
                     arrowprops=dict(arrowstyle="-|>", color="#16a085", lw=2))
    ax_cell.text(0.95, (yA + yB)/2 + 0.3, "la memoria $C$ fluye\ncon poca alteración",
                 fontsize=8, color="#16a085", va="center")
    ax_cell.set_title("Dentro de la celda LSTM", fontsize=12, fontweight="bold")

    # --- C y h en el tiempo ---
    def heat(ax, M, titulo):
        mask = np.zeros_like(M, dtype=bool); mask[:, paso+1:] = True
        cmap = plt.get_cmap(VAL).copy(); cmap.set_bad("#f0f0f0")
        ax.imshow(np.ma.array(M, mask=mask), cmap=cmap, vmin=-1, vmax=1,
                  aspect="auto", origin="upper")
        for r in range(H):
            for c in range(paso + 1):
                ax.text(c, r, f"{M[r, c]:+.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(M[r, c]) > 0.6 else "#222")
        ax.set_xticks(range(T)); ax.set_xticklabels(TOKENS, fontsize=8, rotation=20)
        ax.set_yticks(range(H)); ax.set_yticklabels(range(H), fontsize=8)
        ax.add_patch(mpatches.Rectangle((paso - 0.5, -0.5), 1, H, fill=False,
                     edgecolor="#c0392b", lw=2.5, zorder=5))
        ax.set_title(titulo, fontsize=10.5, fontweight="bold")

    heat(ax_C, S["C"][1:].T, "Estado de celda C en el tiempo (memoria larga)")
    heat(ax_h, S["h"][1:].T, "Estado oculto h en el tiempo (salida)")

    fig.suptitle("Celda LSTM: tres compuertas (olvidar, escribir, exponer) "
                 "regulan una memoria C que fluye por la secuencia",
                 fontsize=12, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def lstm_interactiva():
    """Despliega la LSTM interactiva: procesa la frase palabra por palabra.

    Controles: un slider de paso y un botón para animar la secuencia. En cada
    paso se ven las compuertas (forget, input, output), el candidato, y cómo se
    actualizan el estado de celda C y el estado oculto h.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_lstm(estado["paso"])
            plt.show()

    s_paso = widgets.IntSlider(value=0, min=0, max=T - 1, step=1, description="paso t",
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
                time.sleep(1.2)
            s_paso.unobserve(on_paso, names="value")
            s_paso.value = T - 1
            s_paso.observe(on_paso, names="value")
        finally:
            for w in (b_run, s_paso):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Celda LSTM</h3>"
        "<span style='color:#555'>La LSTM mantiene una <b>memoria de celda C</b> "
        "que fluye por la secuencia con poca alteración. En cada paso, tres "
        "<b>compuertas</b> deciden qué <b>olvidar</b> (f), qué <b>escribir</b> "
        "(i·g) y qué <b>exponer</b> como salida h (o). Avanza con el slider o "
        "pulsa Procesar.</span>")
    controles = widgets.HBox([s_paso, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
