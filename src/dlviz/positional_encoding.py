"""Animación didáctica de la codificación posicional (positional encoding).

La self-attention es invariante al orden: trata la secuencia como un conjunto.
Las RNN tenían el orden "gratis" al procesar paso a paso, pero el Transformer
procesa todo en paralelo, así que el orden se inyecta SUMANDO a cada token un
vector que depende de su posición:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Cada posición recibe un vector único; cada dimensión es una onda de distinta
frecuencia. La entrada al Transformer es embedding (significado) + PE (posición).

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from positional_encoding import positional_encoding_interactiva
    positional_encoding_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


FRASE = ["el", "pequeño", "gato", "negro", "bebe", "la", "leche", "fría",
         "muy", "rápido"]
L = len(FRASE)        # posiciones
DM = 32               # dimensión del modelo
_EMB = np.random.default_rng(5).normal(0, 0.4, (L, DM))   # embeddings ilustrativos
DIMS_CURVA = [0, 2, 8, 20]                                # dims a graficar


def codificacion_posicional(L=L, d=DM):
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / (10000.0 ** (2 * (i // 2) / d))
    PE = np.zeros((L, d))
    PE[:, 0::2] = np.sin(ang[:, 0::2])
    PE[:, 1::2] = np.cos(ang[:, 1::2])
    return PE


PE = codificacion_posicional()


def _fila(ax, x0, y0, vals, cmap, vmin, vmax, cw=1.0, h=0.8):
    cm = plt.get_cmap(cmap)
    for j, v in enumerate(vals):
        t = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
        ax.add_patch(mpatches.Rectangle((x0 + j*cw, y0), cw, h,
                     facecolor=cm(t)[:3], edgecolor="none"))


def figura_pos_encoding(pos=0):
    pos = int(np.clip(pos, 0, L - 1))

    fig = plt.figure(figsize=(13, 7.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.35, 1.0, 1.0], width_ratios=[1.15, 1])
    ax_chips = fig.add_subplot(gs[0, :])
    ax_heat = fig.add_subplot(gs[1:3, 0])
    ax_curve = fig.add_subplot(gs[1, 1])
    ax_add = fig.add_subplot(gs[2, 1])

    # ---- Frase con posiciones ----
    ax_chips.set_xlim(-0.5, L * 1.25); ax_chips.set_ylim(-0.5, 1.2); ax_chips.axis("off")
    for k, tok in enumerate(FRASE):
        x = k * 1.25
        sel = (k == pos)
        ax_chips.add_patch(mpatches.FancyBboxPatch((x - 0.55, 0.05), 1.1, 0.7,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor="#e67e22" if sel else "#eef2f7",
                           edgecolor="#33425b", lw=2.4 if sel else 1.0, zorder=3))
        ax_chips.text(x, 0.4, tok, ha="center", va="center", fontsize=9.5,
                      color="white" if sel else "#222",
                      fontweight="bold" if sel else "normal")
        ax_chips.text(x, -0.3, f"pos {k}", ha="center", fontsize=7.5,
                      color="#e67e22" if sel else "#999")
    ax_chips.set_title("La self-attention no sabe el orden → se suma a cada token "
                       "un vector según su posición", fontsize=11, fontweight="bold")

    # ---- Mapa de calor de la codificación posicional ----
    im = ax_heat.imshow(PE, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto",
                        origin="upper")
    ax_heat.add_patch(mpatches.Rectangle((-0.5, pos - 0.5), DM, 1, fill=False,
                      edgecolor="#e67e22", lw=3, zorder=5))
    ax_heat.set_yticks(range(L)); ax_heat.set_yticklabels(
        [f"{k}  {t}" for k, t in enumerate(FRASE)], fontsize=8)
    ax_heat.set_xlabel("dimensión del embedding", fontsize=9.5)
    ax_heat.set_ylabel("posición", fontsize=9.5)
    ax_heat.set_title("Codificación posicional: cada posición → un vector único\n"
                      "(columnas = ondas de distinta frecuencia)",
                      fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # ---- Ondas por dimensión ----
    xx = np.arange(L)
    cmap_c = plt.get_cmap("viridis")
    for idx, dim in enumerate(DIMS_CURVA):
        ax_curve.plot(xx, PE[:, dim], "o-", ms=3, lw=1.8,
                      color=cmap_c(idx / max(len(DIMS_CURVA)-1, 1)),
                      label=f"dim {dim}")
    ax_curve.axvline(pos, color="#e67e22", lw=2, ls="--")
    ax_curve.set_xlabel("posición", fontsize=9); ax_curve.set_ylabel("valor", fontsize=9)
    ax_curve.set_ylim(-1.25, 1.25)
    ax_curve.legend(fontsize=8, ncol=2, loc="lower right")
    ax_curve.grid(alpha=0.3)
    ax_curve.set_title("Cada dimensión oscila a distinta frecuencia\n"
                       "(dims bajas: rápidas · dims altas: lentas)",
                       fontsize=10, fontweight="bold")

    # ---- Uso: embedding + PE = entrada ----
    emb = _EMB[pos]
    suma = emb + PE[pos]
    M = np.vstack([emb, PE[pos], suma])
    vmax = np.abs(M).max()
    ax_add.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto",
                  origin="upper")
    ax_add.set_yticks([0, 1, 2])
    ax_add.set_yticklabels(["embedding\n(significado)", f"PE(pos {pos})\n(posición)",
                            "entrada"], fontsize=8.5)
    ax_add.set_xlabel("dimensión", fontsize=9)
    ax_add.set_title(f"Entrada del token «{FRASE[pos]}» = embedding + PE(pos {pos})",
                     fontsize=10, fontweight="bold")
    for y in (0.5, 1.5):
        ax_add.axhline(y, color="white", lw=2)
    ax_add.text(-0.05, 0.667, "+", transform=ax_add.transAxes, fontsize=15,
                ha="center", va="center", fontweight="bold")
    ax_add.text(-0.05, 0.333, "=", transform=ax_add.transAxes, fontsize=15,
                ha="center", va="center", fontweight="bold")

    fig.suptitle(r"Positional encoding: inyectar el orden que la self-attention "
                 r"no tiene  ·  $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$, "
                 r"$PE_{(pos,2i+1)}=\cos(\cdot)$", fontsize=11.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def positional_encoding_interactiva():
    """Despliega la codificación posicional interactiva.

    Controles: un slider de posición (resalta esa fila en el mapa de calor, la
    marca en las ondas y muestra la suma embedding + PE) y un botón para recorrer
    la frase posición por posición.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"pos": 0}
    out = widgets.Output(layout=widgets.Layout(height="740px", overflow="hidden"))

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_pos_encoding(estado["pos"])
            plt.show()

    s_pos = widgets.IntSlider(value=0, min=0, max=L - 1, step=1, description="posición",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="460px"),
                              continuous_update=False)
    b_run = widgets.Button(description="▶ Recorrer posiciones", button_style="success")

    def on_pos(_):
        estado["pos"] = s_pos.value
        redibujar()
    s_pos.observe(on_pos, names="value")

    def on_run(_):
        for w in (b_run, s_pos):
            w.disabled = True
        try:
            for k in range(L):
                estado["pos"] = k
                redibujar()
                time.sleep(0.8)
            s_pos.unobserve(on_pos, names="value")
            s_pos.value = L - 1
            s_pos.observe(on_pos, names="value")
        finally:
            for w in (b_run, s_pos):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Codificación posicional</h3>"
        "<span style='color:#555'>La self-attention trata la frase como un "
        "<b>conjunto</b> (no sabe el orden). Se suma a cada token un <b>vector de "
        "posición</b> hecho de senos y cosenos de distintas frecuencias, así que "
        "cada posición queda con una firma única. Mueve el slider para ver la "
        "posición elegida.</span>")
    controles = widgets.HBox([s_pos, b_run])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
