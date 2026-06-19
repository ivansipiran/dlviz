"""Animación didáctica de una red encoder-decoder (seq2seq) para traducción.

Reproduce el esquema clásico de Neural Machine Translation:

  * El ENCODER (RNN) lee la frase origen palabra por palabra y resume toda la
    secuencia en su último estado oculto: el VECTOR DE CONTEXTO.
  * Ese vector pasa al DECODER (RNN), que genera la traducción palabra por
    palabra, partiendo de <START> y realimentando cada palabra generada como
    entrada del paso siguiente, hasta producir <END>.

El cuello de botella (toda la frase comprimida en un vector fijo) motiva, más
adelante, la idea de atención.

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from seq2seq import seq2seq_interactiva
    seq2seq_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ENC = ["El", "bebé", "está", "llorando"]
DEC_IN = ["<START>", "The", "baby", "is", "crying"]
DEC_OUT = ["The", "baby", "is", "crying", "<END>"]

EX = [1.2, 2.4, 3.6, 4.8]                       # x de celdas del encoder
DX = [7.2, 8.4, 9.6, 10.8, 12.0]                # x de celdas del decoder
YC = 2.0                                        # y de las celdas
YW = 0.75                                       # y de palabras de entrada
YO = 3.45                                       # y de palabras de salida

C_ENC = "#283593"      # azul encoder
C_DEC = "#1e8449"      # verde decoder
C_ACT = "#e67e22"      # naranjo: celda activa
C_GRIS = "#cfd5dc"
C_CTX = "#2980b9"      # azul vector de contexto
C_FB = "#e1a100"       # amarillo realimentación

N_FRAMES = len(ENC) + 1 + len(DEC_OUT)          # encoder + contexto + decoder


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _celda(ax, x, color_tema, revelada, activa):
    edge = C_ACT if activa else (color_tema if revelada else C_GRIS)
    face = "#fdecea" if activa else "white"
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.26, YC - 0.6), 0.52, 1.2,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=face, edgecolor=edge, lw=3 if activa else 1.8, zorder=3))
    cdot = color_tema if revelada else "#e3e7ec"
    for dy in (0.36, 0.12, -0.12, -0.36):
        ax.add_patch(mpatches.Circle((x, YC + dy), 0.07, facecolor=cdot,
                     edgecolor="none", zorder=4))


def _flecha(ax, p0, p1, color, lw=2, ls="-", alpha=1.0):
    ax.annotate("", xy=p1, xytext=p0, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=alpha))


def figura_seq2seq(frame=0):
    frame = int(np.clip(frame, 0, N_FRAMES - 1))
    n_enc = len(ENC)
    if frame < n_enc:
        fase, enc_act, ctx, dec_rev, dec_act = "enc", frame, False, 0, None
    elif frame == n_enc:
        fase, enc_act, ctx, dec_rev, dec_act = "ctx", None, True, 0, None
    else:
        d = frame - n_enc - 1
        fase, enc_act, ctx, dec_rev, dec_act = "dec", None, True, d + 1, d
    enc_rev = n_enc if frame >= n_enc else frame + 1

    fig = plt.figure(figsize=(14, 7.2), layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.3, 13.2); ax.set_ylim(-1.9, 5.0); ax.axis("off")

    # ---- ENCODER ----
    ax.text(EX[0] - 0.9, YC, "Encoder\nRNN", ha="right", va="center",
            fontsize=11, fontweight="bold", color=C_ENC)
    for e in range(n_enc):
        rev = e < enc_rev
        act = (e == enc_act)
        _celda(ax, EX[e], C_ENC, rev, act)
        # palabra origen -> celda
        col = C_ACT if act else (C_ENC if rev else C_GRIS)
        _flecha(ax, (EX[e], YW + 0.25), (EX[e], YC - 0.62),
                col if rev else C_GRIS, lw=1.8)
        ax.text(EX[e], YW, ENC[e], ha="center", va="center", fontsize=11,
                color="#222" if rev else "#aaa")
        if e > 0:  # recurrencia
            _flecha(ax, (EX[e-1] + 0.28, YC), (EX[e] - 0.28, YC),
                    C_ENC if rev else C_GRIS, lw=2)
    ax.text((EX[0] + EX[-1]) / 2, YW - 1.0, "Sentencia origen (input)",
            ha="center", fontsize=10.5, color="#444")

    # caja del vector de contexto en la última celda del encoder
    if ctx:
        ax.add_patch(mpatches.Rectangle((EX[-1] - 0.34, YC - 0.7), 0.68, 1.4,
                     fill=False, edgecolor=C_CTX, lw=2.6, zorder=6))
        ax.annotate("vector de contexto\n(resume toda la frase)",
                    xy=(EX[-1], YC + 0.72), xytext=(EX[-1] - 0.2, 4.5),
                    ha="center", fontsize=9.5, color=C_CTX, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=C_CTX, lw=1.8))

    # ---- PUENTE encoder -> decoder ----
    _flecha(ax, (EX[-1] + 0.3, YC), (DX[0] - 0.3, YC),
            C_CTX if ctx else C_GRIS, lw=2.6 if ctx else 1.5,
            alpha=1.0 if ctx else 0.5)

    # ---- DECODER ----
    ax.text(DX[-1] + 0.9, YC, "Decoder\nRNN", ha="left", va="center",
            fontsize=11, fontweight="bold", color=C_DEC)
    for d in range(len(DEC_OUT)):
        rev = d < dec_rev
        act = (d == dec_act)
        _celda(ax, DX[d], C_DEC, rev, act)
        col = C_ACT if act else (C_DEC if rev else C_GRIS)
        # entrada (palabra previa o <START>) -> celda
        _flecha(ax, (DX[d], YW + 0.25), (DX[d], YC - 0.62), col if rev else C_GRIS, lw=1.8)
        ax.text(DX[d], YW, DEC_IN[d], ha="center", va="center", fontsize=10.5,
                color="#222" if rev else "#aaa")
        # celda -> salida (argmax)
        _flecha(ax, (DX[d], YC + 0.62), (DX[d], YO - 0.2), col if rev else C_GRIS, lw=1.8)
        ax.text(DX[d] - 0.16, (YC + 0.62 + YO) / 2, "argmax", rotation=90,
                ha="center", va="center", fontsize=7, color="#999")
        ax.text(DX[d], YO, DEC_OUT[d], ha="center", va="center", fontsize=11,
                fontweight="bold" if rev else "normal",
                color="#222" if rev else "#bbb")
        if d > 0:  # recurrencia decoder
            _flecha(ax, (DX[d-1] + 0.28, YC), (DX[d] - 0.28, YC),
                    C_DEC if rev else C_GRIS, lw=2)
        # realimentación: salida d-1 -> entrada d
        if d >= 1 and d < dec_rev:
            _flecha(ax, (DX[d-1] + 0.18, YO - 0.05), (DX[d] - 0.05, YW + 0.45),
                    C_FB, lw=1.8, ls="--")
    ax.text((DX[0] + DX[-1]) / 2, YO + 0.7, "Secuencia target (salida)",
            ha="center", fontsize=10.5, color="#444")

    # flecha de salida final del decoder
    _flecha(ax, (DX[-1] + 0.3, YC), (DX[-1] + 0.7, YC),
            C_DEC if fase == "dec" else C_GRIS, lw=2)

    # ---- Cartel motivador (cuello de botella -> atención) ----
    if ctx:
        ax.text(6.0, -1.55,
                "Toda la frase origen se comprime en UN vector fijo (cuello de "
                "botella).  →  La atención permitirá al decoder mirar todos los "
                "estados del encoder.", ha="center", fontsize=9.5, color=C_CTX,
                bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2fb", ec=C_CTX))

    # ---- Caption del paso ----
    if fase == "enc":
        cap = f"Paso del encoder {frame+1}/{n_enc}: lee «{ENC[frame]}» y actualiza su estado oculto."
    elif fase == "ctx":
        cap = "El último estado del encoder es el vector de contexto: resume toda la frase origen."
    else:
        d = dec_act
        if d == 0:
            cap = "El decoder arranca con «<START>» + el contexto, y genera «The» (argmax sobre el vocabulario)."
        elif DEC_OUT[d] == "<END>":
            cap = "Genera «<END>»: la traducción termina."
        else:
            cap = (f"Genera «{DEC_OUT[d]}» a partir del contexto y de la palabra previa "
                   f"«{DEC_IN[d]}» (realimentación, flecha amarilla).")
    fig.suptitle("Traducción con encoder-decoder (seq2seq)\n" + cap,
                 fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def seq2seq_interactiva():
    """Despliega la animación del encoder-decoder para traducción.

    Controles: un slider de paso y un botón para reproducir toda la secuencia
    (primero el encoder lee la frase origen, luego el decoder genera la
    traducción palabra por palabra).
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"frame": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_seq2seq(estado["frame"])
            plt.show()

    s_frame = widgets.IntSlider(value=0, min=0, max=N_FRAMES - 1, step=1,
                                description="paso",
                                style={"description_width": "50px"},
                                layout=widgets.Layout(width="460px"),
                                continuous_update=False)
    b_run = widgets.Button(description="▶ Traducir", button_style="success")

    def on_frame(_):
        estado["frame"] = s_frame.value
        redibujar()
    s_frame.observe(on_frame, names="value")

    def on_run(_):
        for w in (b_run, s_frame):
            w.disabled = True
        try:
            for k in range(N_FRAMES):
                estado["frame"] = k
                redibujar()
                time.sleep(1.1)
            s_frame.unobserve(on_frame, names="value")
            s_frame.value = N_FRAMES - 1
            s_frame.observe(on_frame, names="value")
        finally:
            for w in (b_run, s_frame):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Encoder-decoder para traducción</h3>"
        "<span style='color:#555'>El <b>encoder</b> lee la frase origen y la "
        "resume en un <b>vector de contexto</b>; el <b>decoder</b> genera la "
        "traducción palabra por palabra, realimentando cada palabra como entrada "
        "del paso siguiente. Avanza con el slider o pulsa Traducir.</span>")
    controles = widgets.HBox([s_frame, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
