"""Animación didáctica de canales y filtros múltiples en convolución.

Explica dos ideas clave de las CNN sobre imágenes RGB:

1. Un filtro abarca TODOS los canales de entrada: un "filtro 3×3 sobre RGB" es
   en realidad 3×3×3. En cada posición, cada canal se multiplica por su rebanada
   del filtro y los tres se SUMAN en un solo número.
2. Cada filtro produce UN mapa de activación. Con F filtros se obtiene un volumen
   de salida de profundidad F (varias dimensiones).

Este módulo es autónomo (solo numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from multicanal import multicanal_interactiva
    multicanal_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Imagen RGB de ejemplo y filtros
# ---------------------------------------------------------------------------
def imagen_rgb():
    im = np.zeros((7, 7, 3))
    im[0:3, 0:3, 0] = 1.0       # cuadrado rojo  (arriba-izquierda)
    im[0:3, 4:7, 1] = 1.0       # cuadrado verde (arriba-derecha)
    im[4:7, 2:5, 2] = 1.0       # cuadrado azul  (abajo-centro)
    return im


def _filtro_color(k):
    f = np.full((3, 3, 3), -0.5 / 9.0)
    f[:, :, k] = 1.0 / 9.0
    return f


_LAP = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)

FILTROS = {
    "Rojo": _filtro_color(0),
    "Verde": _filtro_color(1),
    "Azul": _filtro_color(2),
    "Bordes": np.stack([_LAP / 3.0] * 3, axis=-1),
}
NOMBRES = list(FILTROS.keys())
CANALES = ["R", "G", "B"]
CMAP_CANAL = ["Reds", "Greens", "Blues"]


def conv_multicanal(img, f):
    """Convolución de una imagen (H,W,C) con un filtro (k,k,C). Salida 2D."""
    H, W, _ = img.shape
    k = f.shape[0]
    oh, ow = H - k + 1, W - k + 1
    out = np.zeros((oh, ow))
    for r in range(oh):
        for c in range(ow):
            out[r, c] = np.sum(img[r:r + k, c:c + k, :] * f)
    return out


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _ventana(ax, rr, cc, k, color="#e67e22"):
    ax.add_patch(mpatches.Rectangle((cc - 0.5, rr - 0.5), k, k, fill=False,
                 edgecolor=color, lw=3, zorder=5))


def _img_grid(ax, M, cmap, vmin, vmax, titulo, win=None, k=3):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    h, w = M.shape[:2]
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#ccc", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if win is not None:
        _ventana(ax, win[0], win[1], k)
    ax.set_title(titulo, fontsize=9.5, fontweight="bold")


def _slice_filtro(ax, sl, titulo):
    m = np.abs(sl).max() or 1.0
    ax.imshow(sl, cmap="RdBu_r", vmin=-m, vmax=m, origin="upper")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{sl[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=9)


def figura_multicanal(filtro="Rojo", paso=0):
    img = imagen_rgb()
    k = 3
    mapas = {n: conv_multicanal(img, f) for n, f in FILTROS.items()}
    oh, ow = mapas[NOMBRES[0]].shape
    n_out = oh * ow
    paso = int(np.clip(paso, 0, n_out - 1))
    r_o, c_o = divmod(paso, ow)
    rr, cc = r_o, c_o

    fsel = FILTROS[filtro]
    parche = img[rr:rr + k, cc:cc + k, :]
    sumas = [float((parche[:, :, ch] * fsel[:, :, ch]).sum()) for ch in range(3)]
    valor = sum(sumas)

    fig = plt.figure(figsize=(13.5, 8.5), layout="constrained")
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 1.25],
                          height_ratios=[1, 1, 1, 1])

    # Columna 0: entrada (RGB compuesta + 3 canales)
    ax_rgb = fig.add_subplot(gs[0, 0])
    _img_grid(ax_rgb, img, None, 0, 1, "Entrada RGB", win=(rr, cc))
    for ch in range(3):
        axc = fig.add_subplot(gs[ch + 1, 0])
        _img_grid(axc, img[:, :, ch], CMAP_CANAL[ch], 0, 1,
                  f"canal {CANALES[ch]}", win=(rr, cc))

    # Columna 1: filtro seleccionado (3 rebanadas) + cálculo
    for ch in range(3):
        axf = fig.add_subplot(gs[ch, 1])
        _slice_filtro(axf, fsel[:, :, ch], f"filtro «{filtro}» · canal {CANALES[ch]}")
    ax_calc = fig.add_subplot(gs[3, 1])
    ax_calc.axis("off"); ax_calc.set_xlim(0, 1); ax_calc.set_ylim(0, 1)
    cols = ["#c0392b", "#27ae60", "#2980b9"]
    ax_calc.text(0.0, 0.9, "Cada canal × su filtro, y se SUMAN:", fontsize=9.5,
                 fontweight="bold", va="center")
    partes = "   +   ".join([f"$s_{CANALES[ch]}$={sumas[ch]:.2f}" for ch in range(3)])
    ax_calc.text(0.0, 0.6, partes, fontsize=10, va="center")
    ax_calc.text(0.0, 0.28, "=", fontsize=12, va="center")
    ax_calc.text(0.1, 0.28, f"{valor:.2f}", fontsize=16, fontweight="bold",
                 color="#8e44ad", va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f3e9f7", ec="#8e44ad"))
    ax_calc.text(0.0, 0.02, f"→ 1 píxel del mapa «{filtro}»", fontsize=9,
                 color="#555", va="center")

    # Columna 2: volumen de salida (F mapas en mosaico)
    ax_out = fig.add_subplot(gs[:, 2])
    vmin = min(m.min() for m in mapas.values())
    vmax = max(m.max() for m in mapas.values())
    ncol = 2
    big_h, big_w = 2 * oh + 1, ncol * ow + 1
    big = np.full((big_h, big_w), np.nan)
    idx = np.arange(n_out).reshape(oh, ow)
    origenes = {}
    for fi, n in enumerate(NOMBRES):
        tr, tc = divmod(fi, ncol)
        r0, c0 = tr * (oh + 1), tc * (ow + 1)
        mm = np.where(idx <= paso, mapas[n], np.nan)
        big[r0:r0 + oh, c0:c0 + ow] = mm
        origenes[n] = (r0, c0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    ax_out.imshow(np.ma.masked_invalid(big), cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="upper")
    for n in NOMBRES:
        r0, c0 = origenes[n]
        sel = (n == filtro)
        ax_out.add_patch(mpatches.Rectangle((c0 - 0.5, r0 - 0.5), ow, oh, fill=False,
                         edgecolor="#8e44ad" if sel else "#bbb",
                         lw=3 if sel else 1.2, zorder=4))
        ax_out.text(c0 + ow / 2 - 0.5, r0 - 0.8, n, ha="center", fontsize=9,
                    fontweight="bold", color="#8e44ad" if sel else "#555")
        ax_out.add_patch(mpatches.Rectangle((c0 + c_o - 0.5, r0 + r_o - 0.5), 1, 1,
                         fill=False, edgecolor="#c0392b", lw=2, zorder=5))
    ax_out.set_xticks([]); ax_out.set_yticks([])
    ax_out.set_xlim(-0.7, big_w - 0.3)
    ax_out.set_ylim(big_h - 0.5, -1.8)
    ax_out.set_title(f"Salida: volumen de {len(NOMBRES)} mapas  ({oh}×{ow}×"
                     f"{len(NOMBRES)})\nprofundidad de salida = nº de filtros",
                     fontsize=10.5, fontweight="bold", pad=14)

    fig.suptitle("Canales y filtros: un filtro 3×3 sobre RGB es 3×3×3 = 27 pesos; "
                 "cada filtro genera un mapa de activación",
                 fontsize=12, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def multicanal_interactiva(filtro="Rojo"):
    """Despliega la convolución multicanal interactiva sobre una imagen RGB.

    Controles: qué filtro inspeccionar (su rebanada por canal y su cálculo), una
    posición (slider para recorrer la ventana) y un botón para animar el barrido.
    Todos los filtros calculan a la vez: a la derecha se ve crecer el volumen de
    mapas de salida (uno por filtro).
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    oh = ow = 7 - 3 + 1
    n_out = oh * ow
    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_multicanal(s_filtro.value, estado["paso"])
            plt.show()

    s_filtro = widgets.Dropdown(options=NOMBRES, value=filtro,
                                description="filtro",
                                style={"description_width": "55px"},
                                layout=widgets.Layout(width="220px"))
    s_pos = widgets.IntSlider(value=0, min=0, max=n_out - 1, step=1,
                              description="posición",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="430px"),
                              continuous_update=False)
    b_run = widgets.Button(description="▶ Animar barrido", button_style="success")

    def on_pos(_):
        estado["paso"] = s_pos.value
        redibujar()

    def on_filtro(_):
        redibujar()

    s_pos.observe(on_pos, names="value")
    s_filtro.observe(on_filtro, names="value")

    def on_run(_):
        for w in (b_run, s_filtro, s_pos):
            w.disabled = True
        try:
            for k in range(n_out):
                estado["paso"] = k
                redibujar()
                time.sleep(0.12)
            s_pos.unobserve(on_pos, names="value")
            s_pos.value = n_out - 1
            s_pos.observe(on_pos, names="value")
        finally:
            for w in (b_run, s_filtro, s_pos):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Canales y filtros múltiples</h3>"
        "<span style='color:#555'>La imagen RGB tiene 3 canales. Un filtro tiene "
        "una rebanada por canal; en cada posición multiplica cada canal por su "
        "rebanada y <b>suma</b> todo en un número. Cada filtro produce un mapa, y "
        "varios filtros forman el <b>volumen de salida</b>. Elige un filtro y "
        "recorre la imagen.</span>")
    controles = widgets.VBox([widgets.HBox([s_filtro, b_run]), s_pos])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
