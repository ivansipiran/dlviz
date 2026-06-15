"""Animación didáctica de padding y stride en una convolución.

Muestra cómo el padding (borde de ceros alrededor de la imagen) y el stride
(salto de la ventana) cambian el recorrido del kernel y el tamaño del mapa de
salida. Pensado para Colab/Jupyter.

Reutiliza las imágenes y kernels de :mod:`cc6204viz.conv`.

Función de alto nivel: :func:`pad_stride_interactiva`.

Para activarla en el paquete, añade en ``__init__.py``::

    from .pad_stride import pad_stride_interactiva
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .conv import IMAGENES, KERNELS


# ---------------------------------------------------------------------------
# Cómputo
# ---------------------------------------------------------------------------
def convolucionar_ps(img, K, padding=0, stride=1):
    """Convolución (correlación) con padding de ceros y stride.

    Devuelve (salida, imagen_con_padding, (oh, ow)).
    """
    kh, kw = K.shape
    imgp = np.pad(img, padding, mode="constant") if padding > 0 else img.copy()
    H, W = imgp.shape
    oh = (H - kh) // stride + 1
    ow = (W - kw) // stride + 1
    out = np.zeros((oh, ow))
    for r in range(oh):
        for c in range(ow):
            rr, cc = r * stride, c * stride
            out[r, c] = np.sum(imgp[rr:rr + kh, cc:cc + kw] * K)
    return out, imgp, (oh, ow)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _dibujar_padded(ax, imgp, padding, orig_shape, kh, kw, win_rc, centros, cur_idx):
    H, W = imgp.shape
    Ho, Wo = orig_shape
    ax.imshow(imgp, cmap="gray", vmin=0, vmax=1, origin="upper")

    # Celdas de padding resaltadas en azul translúcido
    for i in range(H):
        for j in range(W):
            es_pad = (i < padding or i >= padding + Ho or
                      j < padding or j >= padding + Wo)
            if es_pad:
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#3498db", alpha=0.35, zorder=2))

    # Líneas de grilla
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#bbb", lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])

    # Borde de la imagen original
    if padding > 0:
        ax.add_patch(mpatches.Rectangle((padding - 0.5, padding - 0.5), Wo, Ho,
                     fill=False, edgecolor="white", lw=1.6, ls="--", zorder=3))

    # Retícula de centros de ventana (muestra el efecto del stride)
    cs = np.array(centros)
    ax.plot(cs[:, 0], cs[:, 1], "o", ms=4, color="#555", alpha=0.55, zorder=4)
    ax.plot([cs[cur_idx, 0]], [cs[cur_idx, 1]], "o", ms=7, color="#c0392b", zorder=6)

    # Ventana actual
    rr, cc = win_rc
    ax.add_patch(mpatches.Rectangle((cc - 0.5, rr - 0.5), kw, kh, fill=False,
                 edgecolor="#e67e22", lw=3, zorder=5))
    ax.set_title(f"Entrada con padding  ({H}×{W})", fontsize=11, fontweight="bold")


def _dibujar_salida(ax, out, oh, ow, cur_rc, mascara):
    disp = np.ma.array(out, mask=mascara)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(disp, cmap=cmap, vmin=out.min(), vmax=out.max(), origin="upper")
    ax.set_xticks(np.arange(-0.5, ow, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, oh, 1), minor=True)
    ax.grid(which="minor", color="#bbb", lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    r, c = cur_rc
    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                 edgecolor="#c0392b", lw=3, zorder=5))
    ax.set_title(f"Mapa de salida  ({oh}×{ow})", fontsize=11, fontweight="bold")


def figura_pad_stride(img_key="Cuadrado", kernel_key="Sobel vertical",
                      padding=1, stride=1, paso=0):
    img = IMAGENES[img_key]()
    K = KERNELS[kernel_key]
    kh, kw = K.shape
    Ho, Wo = img.shape

    out, imgp, (oh, ow) = convolucionar_ps(img, K, padding, stride)
    n_out = oh * ow
    paso = int(np.clip(paso, 0, n_out - 1))
    r_o, c_o = divmod(paso, ow)
    rr, cc = r_o * stride, c_o * stride
    valor = out[r_o, c_o]

    # Centros de cada posición de ventana (en coords de imagen con padding)
    centros = [((c % ow) * stride + (kw - 1) / 2, (c // ow) * stride + (kh - 1) / 2)
               for c in range(n_out)]

    idx = np.arange(n_out).reshape(oh, ow)
    mascara = idx > paso

    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.3, 0.7])
    ax_in = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    _dibujar_padded(ax_in, imgp, padding, (Ho, Wo), kh, kw, (rr, cc), centros, paso)
    _dibujar_salida(ax_out, out, oh, ow, (r_o, c_o), mascara)

    # Panel de información: fórmula + números + notas
    ax_info.axis("off")
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    formula = (r"$o=\left\lfloor\dfrac{H+2P-K}{S}\right\rfloor+1"
               r"=\left\lfloor\dfrac{%d+2\cdot%d-%d}{%d}\right\rfloor+1=%d$"
               % (Ho, padding, kh, stride, oh))
    ax_info.text(0.0, 0.86, formula, fontsize=15, va="center")

    mismo = " (= entrada, 'same')" if (oh, ow) == (Ho, Wo) else ""
    ax_info.text(0.0, 0.52,
                 f"Padding P = {padding}: añade un borde de ceros (azul). "
                 f"Permite que los píxeles del borde participen y agranda la salida.",
                 fontsize=10, color="#333", va="center")
    ax_info.text(0.0, 0.33,
                 f"Stride S = {stride}: la ventana salta {stride} celda(s) "
                 f"(puntos de la retícula). S>1 submuestrea → salida más pequeña.",
                 fontsize=10, color="#333", va="center")
    ax_info.text(0.0, 0.12,
                 f"Tamaño de salida: {oh}×{ow}{mismo}   ·   "
                 f"posición {paso+1}/{n_out}  →  Σ(ventana × kernel) = {valor:.2f}",
                 fontsize=10.5, color="#c0392b", fontweight="bold", va="center")

    fig.suptitle(f"Padding y stride: «{img_key}» con kernel «{kernel_key}» "
                 f"{kh}×{kw}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def pad_stride_interactiva(img="Cuadrado", kernel="Sobel vertical"):
    """Despliega la convolución interactiva con controles de padding y stride.

    Controles: imagen, kernel, padding (0–3), stride (1–3), una posición (slider
    para recorrer la ventana) y un botón para animar el barrido. El tamaño de la
    salida se recalcula con la fórmula o = ⌊(H+2P−K)/S⌋ + 1 a medida que cambias
    padding y stride.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    def n_posiciones():
        im = IMAGENES[s_img.value]()
        K = KERNELS[s_ker.value]
        kh, kw = K.shape
        H, W = im.shape[0] + 2 * s_pad.value, im.shape[1] + 2 * s_pad.value
        oh = (H - kh) // s_str.value + 1
        ow = (W - kw) // s_str.value + 1
        return oh * ow

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_pad_stride(s_img.value, s_ker.value,
                                    s_pad.value, s_str.value, estado["paso"])
            plt.show()

    s_img = widgets.Dropdown(options=list(IMAGENES.keys()), value=img,
                             description="imagen",
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="220px"))
    s_ker = widgets.Dropdown(options=list(KERNELS.keys()), value=kernel,
                             description="kernel",
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="250px"))
    s_pad = widgets.IntSlider(value=1, min=0, max=3, step=1, description="padding",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="280px"))
    s_str = widgets.IntSlider(value=1, min=1, max=3, step=1, description="stride",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="280px"))
    s_pos = widgets.IntSlider(value=0, min=0, max=n_posiciones() - 1, step=1,
                              description="posición",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="430px"),
                              continuous_update=False)
    b_run = widgets.Button(description="▶ Animar barrido", button_style="success")

    def reset_pos():
        s_pos.unobserve(on_pos, names="value")
        s_pos.max = max(0, n_posiciones() - 1)
        s_pos.value = 0
        s_pos.observe(on_pos, names="value")
        estado["paso"] = 0

    def on_pos(_):
        estado["paso"] = s_pos.value
        redibujar()

    def on_cfg(_):
        reset_pos()
        redibujar()

    s_pos.observe(on_pos, names="value")
    for w in (s_img, s_ker, s_pad, s_str):
        w.observe(on_cfg, names="value")

    def on_run(_):
        for w in (b_run, s_img, s_ker, s_pad, s_str, s_pos):
            w.disabled = True
        try:
            n = n_posiciones()
            for k in range(n):
                estado["paso"] = k
                redibujar()
                time.sleep(0.12)
            s_pos.unobserve(on_pos, names="value")
            s_pos.value = n - 1
            s_pos.observe(on_pos, names="value")
        finally:
            for w in (b_run, s_img, s_ker, s_pad, s_str, s_pos):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Padding y stride</h3>"
        "<span style='color:#555'>El <b>padding</b> rodea la imagen con ceros "
        "(en azul) y el <b>stride</b> es el salto de la ventana. Cambia ambos y "
        "observa cómo varían el recorrido del kernel y el tamaño de la salida.</span>")
    controles = widgets.VBox([
        widgets.HBox([s_img, s_ker, b_run]),
        widgets.HBox([s_pad, s_str]),
        s_pos,
    ])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
