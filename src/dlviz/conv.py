"""Animación didáctica de la convolución de una imagen contra un kernel.

Muestra de forma gráfica cómo un kernel se desliza sobre una imagen simple y va
construyendo el mapa de salida, posición a posición, con la operación
`ventana × kernel → suma` explícita. Permite revisar localidad (cada salida usa
solo una vecindad K×K) y compartición de parámetros (el mismo kernel se aplica
en todas las posiciones). Pensado para Colab/Jupyter.

Función de alto nivel: :func:`conv_interactiva`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Imágenes de ejemplo (9x9, valores en [0,1])
# ---------------------------------------------------------------------------
def _img_cuadrado():
    im = np.zeros((9, 9))
    im[2:7, 2:7] = 1.0
    return im


def _img_borde_vertical():
    im = np.full((9, 9), 0.15)
    im[:, 5:] = 0.9
    return im


def _img_diagonal():
    im = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            im[i, j] = 0.9 if j >= i else 0.1
    return im


def _img_circulo():
    yy, xx = np.mgrid[0:9, 0:9]
    return (((xx - 4) ** 2 + (yy - 4) ** 2) <= 9).astype(float) * 0.85 + 0.05


IMAGENES = {
    "Cuadrado": _img_cuadrado,
    "Borde vertical": _img_borde_vertical,
    "Diagonal": _img_diagonal,
    "Círculo": _img_circulo,
}


# ---------------------------------------------------------------------------
# Kernels de ejemplo (3x3)
# ---------------------------------------------------------------------------
KERNELS = {
    "Identidad": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "Desenfoque": np.ones((3, 3)) / 9.0,
    "Bordes (Laplaciano)": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float),
    "Sobel vertical": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "Sobel horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "Realce (sharpen)": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
}


def convolucionar(img, K):
    """Convolución (correlación) válida, stride 1. Devuelve el mapa de salida."""
    kh, kw = K.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for r in range(oh):
        for c in range(ow):
            out[r, c] = np.sum(img[r:r + kh, c:c + kw] * K)
    return out


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _luminancia_texto(rgb):
    return "white" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5 else "#111"


def _dibujar_grilla(ax, M, titulo, cmap, vmin, vmax, resaltar=None,
                    mascara=None, color_resalte="#e67e22"):
    disp = np.ma.array(M, mask=mascara) if mascara is not None else M
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f2f2f2")
    ax.imshow(disp, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="upper")
    h, w = M.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#bbb", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if resaltar is not None:
        r, c, kh, kw = resaltar
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), kw, kh, fill=False,
                                        edgecolor=color_resalte, lw=3, zorder=5))
    ax.set_title(titulo, fontsize=11, fontweight="bold")


def _minigrilla(ax, ox, oy, vals, cmap, vmin, vmax, titulo, fmt="{:.2f}"):
    cm = plt.get_cmap(cmap)
    n = vals.shape[0]
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            t = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
            rgb = cm(t)[:3]
            ax.add_patch(mpatches.Rectangle((ox + j, oy - i), 1, 1,
                         facecolor=rgb, edgecolor="#666", lw=0.8))
            ax.text(ox + j + 0.5, oy - i + 0.5, fmt.format(v),
                    ha="center", va="center", fontsize=8,
                    color=_luminancia_texto(rgb))
    ax.text(ox + n / 2, oy + 1.3, titulo, ha="center", fontsize=9, fontweight="bold")


def figura_conv(img_key="Cuadrado", kernel_key="Sobel vertical", paso=12):
    img = IMAGENES[img_key]()
    K = KERNELS[kernel_key]
    kh, kw = K.shape
    out = convolucionar(img, K)
    oh, ow = out.shape
    n_out = oh * ow
    paso = int(np.clip(paso, 0, n_out - 1))
    r, c = divmod(paso, ow)

    parche = img[r:r + kh, c:c + kw]
    productos = parche * K
    valor = productos.sum()

    # máscara de salida: celdas aún no calculadas
    idx = np.arange(n_out).reshape(oh, ow)
    mascara_out = idx > paso

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.25, 1])
    ax_in = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, :])

    _dibujar_grilla(ax_in, img, f"Entrada {img.shape[0]}×{img.shape[1]}",
                    "gray", 0, 1, resaltar=(r, c, kh, kw))
    vmax = max(abs(out.min()), abs(out.max()), 1e-6)
    _dibujar_grilla(ax_out, out, f"Mapa de salida {oh}×{ow}",
                    "gray", out.min(), out.max(), resaltar=(r, c, 1, 1),
                    mascara=mascara_out)

    # Tira de cómputo: parche × kernel = productos -> suma
    ax_comp.set_xlim(-0.5, 17.5)
    ax_comp.set_ylim(-2.6, 4.2)
    ax_comp.set_aspect("equal")
    ax_comp.axis("off")
    _minigrilla(ax_comp, 0, 2, parche, "gray", 0, 1, "ventana (entrada)")
    ax_comp.text(3.9, 2.0, "×", fontsize=20, ha="center", va="center")
    _minigrilla(ax_comp, 4.8, 2, K, "RdBu_r", -np.abs(K).max(), np.abs(K).max(), "kernel")
    ax_comp.text(8.7, 2.0, "=", fontsize=20, ha="center", va="center")
    pmax = np.abs(productos).max() or 1.0
    _minigrilla(ax_comp, 9.6, 2, productos, "RdBu_r", -pmax, pmax, "productos")
    ax_comp.text(13.6, 2.0, "→  Σ =", fontsize=15, ha="left", va="center")
    ax_comp.text(15.9, 2.0, f"{valor:.2f}", fontsize=17, ha="center", va="center",
                 fontweight="bold", color="#c0392b",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec="#c0392b"))

    nota = (f"Localidad: la celda resaltada de la salida usa SOLO la ventana "
            f"{kh}×{kw} resaltada en la entrada.\n"
            f"Compartición de parámetros: el MISMO kernel ({kh*kw} números) se "
            f"aplica en las {n_out} posiciones. Una capa densa equivalente "
            f"usaría {img.size}×{n_out} = {img.size*n_out} pesos.")
    ax_comp.text(0, -1.7, nota, ha="left", va="center", fontsize=9.5, color="#333")

    fig.suptitle(f"Convolución: «{img_key}» ✶ «{kernel_key}»      ·      "
                 f"posición {paso+1} de {n_out}  (fila {r}, col {c})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def conv_interactiva(img="Cuadrado", kernel="Sobel vertical"):
    """Despliega la convolución interactiva: kernel deslizándose sobre la imagen.

    Controles: imagen de entrada, kernel, una posición (slider para recorrer a
    mano la ventana) y un botón para animar el barrido completo. La operación
    `ventana × kernel → suma` se muestra abajo en cada posición.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    def n_posiciones(img_key, kernel_key):
        im = IMAGENES[img_key]()
        K = KERNELS[kernel_key]
        return (im.shape[0] - K.shape[0] + 1) * (im.shape[1] - K.shape[1] + 1)

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_conv(s_img.value, s_ker.value, estado["paso"])
            plt.show()

    s_img = widgets.Dropdown(options=list(IMAGENES.keys()), value=img,
                             description="imagen",
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="240px"))
    s_ker = widgets.Dropdown(options=list(KERNELS.keys()), value=kernel,
                             description="kernel",
                             style={"description_width": "60px"},
                             layout=widgets.Layout(width="260px"))
    s_pos = widgets.IntSlider(value=0, min=0, max=n_posiciones(img, kernel) - 1,
                              step=1, description="posición",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="420px"),
                              continuous_update=False)
    b_run = widgets.Button(description="▶ Animar barrido", button_style="success")

    def reset_pos():
        s_pos.unobserve(on_pos, names="value")
        s_pos.max = n_posiciones(s_img.value, s_ker.value) - 1
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
    s_img.observe(on_cfg, names="value")
    s_ker.observe(on_cfg, names="value")

    def on_run(_):
        for w in (b_run, s_img, s_ker, s_pos):
            w.disabled = True
        try:
            n = n_posiciones(s_img.value, s_ker.value)
            for k in range(n):
                estado["paso"] = k
                redibujar()
                time.sleep(0.1)
            s_pos.unobserve(on_pos, names="value")
            s_pos.value = n - 1
            s_pos.observe(on_pos, names="value")
        finally:
            for w in (b_run, s_img, s_ker, s_pos):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Convolución: un kernel que se desliza</h3>"
        "<span style='color:#555'>El kernel recorre la imagen; en cada posición "
        "multiplica su ventana por los pesos y suma, produciendo un píxel de "
        "salida. Recorre con el slider o pulsa Animar y observa cómo se llena el "
        "mapa de salida.</span>")
    controles = widgets.VBox([
        widgets.HBox([s_img, s_ker, b_run]),
        s_pos,
    ])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
