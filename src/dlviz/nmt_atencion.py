"""Animación didáctica de traducción (NMT) con atención.

Continúa el esquema encoder-decoder, pero en lugar de comprimir la frase origen
en un único vector de contexto, en CADA paso del decoder se calcula atención
sobre TODOS los estados del encoder:

  1. El estado del decoder consulta cada estado del encoder  ->  scores.
  2. softmax(scores)  ->  distribución de atención (pesos que suman 1).
  3. salida de atención = suma ponderada de los estados del encoder.
  4. con ella el decoder genera la palabra de salida.

Lo interesante: la atención se concentra (como un foco) en la palabra origen
relevante de cada paso, revelando la alineación (crying <-> llorando, etc.).

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from nmt_atencion import nmt_atencion_interactiva
    nmt_atencion_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


ENC = ["El", "bebé", "está", "llorando"]
DEC_IN = ["<START>", "The", "baby", "is", "crying"]
DEC_OUT = ["The", "baby", "is", "crying", "<END>"]

# Scores de atención (target x origen), diseñados para una alineación clara
_SCORES = np.array([
    [3.0, 0.6, 0.1, 0.0],
    [0.5, 3.0, 0.6, 0.1],
    [0.0, 0.6, 3.0, 1.0],
    [0.0, 0.1, 0.6, 3.0],
    [0.0, 0.0, 0.4, 2.6],
])


def pesos_atencion():
    e = np.exp(_SCORES - _SCORES.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


A = pesos_atencion()
COLS = ["#c0392b", "#e67e22", "#16a085", "#2980b9", "#8e44ad"]  # color por paso

# Geometría
EX = [0.9, 1.9, 2.9, 3.9]      # x de celdas del encoder
Y_ENC = 1.15
Y_WORD = 0.25
Y_SC = 2.25                    # círculos de score
Y_BASE = 2.75                  # base de las barras de distribución
X_DEC = 6.4
X_CTX = 2.4
Y_CTX = 4.45
Y_OUT = 4.45


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _celda(ax, x, y, color, revelada=True, activa=False, w=0.42):
    edge = "#e67e22" if activa else (color if revelada else "#cfd5dc")
    face = "#fdecea" if activa else "white"
    ax.add_patch(mpatches.FancyBboxPatch((x - w/2, y - 0.55), w, 1.1,
                 boxstyle="round,pad=0.02,rounding_size=0.07",
                 facecolor=face, edgecolor=edge, lw=3 if activa else 1.8, zorder=3))
    cd = color if revelada else "#e3e7ec"
    for dy in (0.34, 0.11, -0.11, -0.34):
        ax.add_patch(mpatches.Circle((x, y + dy), 0.062, facecolor=cd,
                     edgecolor="none", zorder=4))


def _arrow(ax, p0, p1, color, lw=1.4, ls="-", alpha=1.0):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=alpha))


def figura_nmt(paso=0):
    paso = int(np.clip(paso, 0, len(DEC_OUT) - 1))
    pesos = A[paso]
    jmax = int(pesos.argmax())
    col = COLS[paso]

    fig = plt.figure(figsize=(14, 7.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax_al = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-0.3, 7.6); ax.set_ylim(-0.5, 5.4); ax.axis("off")

    # ---- Encoder ----
    ax.text(EX[0] - 0.75, Y_ENC, "Encoder\nRNN", ha="right", va="center",
            fontsize=10.5, fontweight="bold", color="#283593")
    for j, w in enumerate(ENC):
        foco = (j == jmax)
        if foco:   # foco sobre la palabra más atendida
            ax.add_patch(mpatches.Rectangle((EX[j] - 0.42, Y_WORD - 0.28), 0.84, 0.5,
                         facecolor=col, alpha=0.18, edgecolor=col, lw=1.5, zorder=1))
        _celda(ax, EX[j], Y_ENC, "#283593")
        _arrow(ax, (EX[j], Y_WORD + 0.18), (EX[j], Y_ENC - 0.57), "#888")
        ax.text(EX[j], Y_WORD, w, ha="center", va="center", fontsize=10.5,
                color=col if foco else "#222",
                fontweight="bold" if foco else "normal")
        if j > 0:
            _arrow(ax, (EX[j-1] + 0.22, Y_ENC), (EX[j] - 0.22, Y_ENC), "#9aa0a6", lw=1.8)
    ax.text((EX[0] + EX[-1]) / 2, Y_WORD - 0.55, "Sentencia origen (input)",
            ha="center", fontsize=9.5, color="#444")

    # ---- Decoder (celda actual) ----
    ax.text(X_DEC + 0.55, Y_ENC, "Decoder\nRNN", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color="#16a085")
    _arrow(ax, (EX[-1] + 0.22, Y_ENC), (X_DEC - 0.24, Y_ENC), "#cfd5dc", lw=1.8)
    _celda(ax, X_DEC, Y_ENC, "#16a085", activa=True)
    _arrow(ax, (X_DEC, Y_WORD + 0.18), (X_DEC, Y_ENC - 0.57), "#16a085")
    ax.text(X_DEC, Y_WORD, DEC_IN[paso], ha="center", va="center", fontsize=10,
            color="#16a085", fontweight="bold")

    # ---- Attention scores (consulta del decoder a cada estado del encoder) ----
    ax.text(EX[0] - 0.75, Y_SC, "Attention\nscores", ha="right", va="center",
            fontsize=9.5, color="#e67e22")
    for j in range(len(ENC)):
        ax.add_patch(mpatches.Circle((EX[j], Y_SC), 0.12, facecolor="#aed6f1",
                     edgecolor="#2980b9", lw=1.3, zorder=4))
        _arrow(ax, (EX[j], Y_ENC + 0.57), (EX[j], Y_SC - 0.13), "#bbb", lw=1.0)
        # consulta desde el decoder
        ax.annotate("", xy=(EX[j] + 0.13, Y_SC), xytext=(X_DEC - 0.22, Y_ENC + 0.2),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa0a6", lw=1.0,
                                    alpha=0.6), zorder=1)

    # ---- Attention distribution (softmax -> barras) ----
    ax.plot([EX[0] - 0.5, EX[-1] + 0.5], [Y_BASE, Y_BASE], color="#2980b9", lw=2)
    ax.text(EX[0] - 0.75, Y_BASE + 0.45, "Attention\ndistribution", ha="right",
            va="center", fontsize=9.5, color="#e67e22")
    for j in range(len(ENC)):
        hb = pesos[j] * 1.4
        ax.add_patch(mpatches.Rectangle((EX[j] - 0.22, Y_BASE), 0.44, hb,
                     facecolor=col if j == jmax else "#f0b27a",
                     edgecolor="#b9770e", lw=0.8, zorder=3))
        ax.text(EX[j], Y_BASE + hb + 0.12, f"{pesos[j]:.2f}", ha="center",
                fontsize=8, color="#7e5109")
        _arrow(ax, (EX[j], Y_SC + 0.13), (EX[j], Y_BASE - 0.02), "#bbb", lw=1.0)

    # ---- Salida de atención = suma ponderada ----
    _celda(ax, X_CTX, Y_CTX, "#d4ac0d", w=0.42)
    for j in range(len(ENC)):
        ax.plot([EX[j], X_CTX], [Y_BASE + pesos[j]*1.4, Y_CTX - 0.5], ls="--",
                color="#bbb", lw=0.9, zorder=1)
    ax.annotate("salida de atención\n(Σ pesos × estados del encoder)",
                xy=(X_CTX + 0.25, Y_CTX), xytext=(X_CTX + 0.7, Y_CTX + 0.55),
                fontsize=9, color="#9a7d0a", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#d4ac0d", lw=1.2))

    # ---- Palabra de salida generada + arco de alineación ----
    ax.add_patch(mpatches.FancyBboxPatch((X_DEC - 0.55, Y_OUT - 0.28), 1.1, 0.56,
                 boxstyle="round,pad=0.04,rounding_size=0.08",
                 facecolor=col, edgecolor="#222", lw=1.5, zorder=4))
    ax.text(X_DEC, Y_OUT, DEC_OUT[paso], ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", zorder=5)
    _arrow(ax, (X_CTX + 0.25, Y_CTX), (X_DEC - 0.56, Y_OUT), col, lw=1.6, ls="--")
    _arrow(ax, (X_DEC, Y_ENC + 0.57), (X_DEC, Y_OUT - 0.3), col, lw=1.6)
    # arco creativo: la palabra generada "viene de" la palabra origen más atendida
    arco = FancyArrowPatch((X_DEC - 0.5, Y_OUT - 0.05), (EX[jmax], Y_WORD + 0.28),
                           connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
                           color=col, lw=2, alpha=0.8, zorder=6)
    ax.add_patch(arco)

    ax.set_title("Atención en la traducción", fontsize=12.5, fontweight="bold")

    # ---- Matriz de alineación (se construye fila por fila) ----
    M = A.copy()
    mask = np.zeros_like(M, dtype=bool); mask[paso + 1:, :] = True
    cmap = plt.get_cmap("Oranges").copy(); cmap.set_bad("#f5f5f5")
    ax_al.imshow(np.ma.array(M, mask=mask), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r in range(paso + 1):
        for c in range(len(ENC)):
            ax_al.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center",
                       fontsize=8.5, color="white" if M[r, c] > 0.5 else "#333")
    ax_al.set_xticks(range(len(ENC))); ax_al.set_xticklabels(ENC, fontsize=9, rotation=20)
    ax_al.set_yticks(range(len(DEC_OUT))); ax_al.set_yticklabels(DEC_OUT, fontsize=9)
    ax_al.set_xlabel("origen", fontsize=9.5); ax_al.set_ylabel("traducción", fontsize=9.5)
    ax_al.add_patch(mpatches.Rectangle((-0.5, paso - 0.5), len(ENC), 1, fill=False,
                    edgecolor=col, lw=2.6, zorder=5))
    ax_al.set_title("Matriz de alineación (¿qué palabra origen\nmira cada palabra "
                    "traducida?)", fontsize=10, fontweight="bold")

    parcial = " ".join(DEC_OUT[:paso]) + "  [" + DEC_OUT[paso] + "]"
    fig.suptitle(f"NMT con atención  ·  generando «{DEC_OUT[paso]}» "
                 f"(mira sobre todo «{ENC[jmax]}»)\nTraducción: {parcial}",
                 fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def nmt_atencion_interactiva():
    """Despliega la traducción con atención, paso a paso.

    En cada paso, el decoder calcula atención sobre todos los estados del
    encoder: se ve la distribución de atención (barras), el foco sobre la palabra
    origen relevante, la salida de atención (suma ponderada) y cómo se va armando
    la matriz de alineación. Controles: slider de paso y botón para reproducir.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_nmt(estado["paso"])
            plt.show()

    s_paso = widgets.IntSlider(value=0, min=0, max=len(DEC_OUT) - 1, step=1,
                               description="paso",
                               style={"description_width": "50px"},
                               layout=widgets.Layout(width="460px"),
                               continuous_update=False)
    b_run = widgets.Button(description="▶ Traducir con atención", button_style="success")

    def on_paso(_):
        estado["paso"] = s_paso.value
        redibujar()
    s_paso.observe(on_paso, names="value")

    def on_run(_):
        for w in (b_run, s_paso):
            w.disabled = True
        try:
            for k in range(len(DEC_OUT)):
                estado["paso"] = k
                redibujar()
                time.sleep(1.3)
            s_paso.unobserve(on_paso, names="value")
            s_paso.value = len(DEC_OUT) - 1
            s_paso.observe(on_paso, names="value")
        finally:
            for w in (b_run, s_paso):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Traducción con atención</h3>"
        "<span style='color:#555'>Ya no hay un solo vector de contexto: en "
        "<b>cada paso</b> el decoder <b>mira toda</b> la frase origen y pondera "
        "cada palabra (atención). El foco de color marca la palabra origen más "
        "atendida; a la derecha se arma la <b>matriz de alineación</b>. Avanza "
        "con el slider o pulsa Traducir.</span>")
    controles = widgets.HBox([s_paso, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
