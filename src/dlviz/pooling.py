"""Animación didáctica de pooling (max y average).

Muestra cómo una ventana de pooling recorre un mapa de activación y lo reduce,
tomando el máximo (max pooling) o el promedio (average pooling) de cada región.
En max pooling se resalta qué celda "gana". Pensado para Colab/Jupyter.

A diferencia de la convolución, el pooling NO tiene parámetros que aprender: es
una operación fija de agregación que submuestrea el mapa.

Este módulo es autónomo (solo numpy / matplotlib / ipywidgets).

Uso en Colab (sube este archivo a /content)::

    from pooling import pooling_interactiva
    pooling_interactiva()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Mapas de ejemplo (8x8, valores en [0,1])
# ---------------------------------------------------------------------------
def _mapa_activacion():
    yy, xx = np.mgrid[0:8, 0:8]
    m = (np.exp(-((xx - 2.3) ** 2 + (yy - 2.3) ** 2) / 4.0)
         + 0.7 * np.exp(-((xx - 5.6) ** 2 + (yy - 5.3) ** 2) / 3.0))
    return m / m.max()


def _mapa_cuadrado():
    m = np.zeros((8, 8))
    m[2:6, 2:6] = 1.0
    return m


def _mapa_diagonal():
    yy, xx = np.mgrid[0:8, 0:8]
    return np.clip((xx - yy + 7) / 14.0, 0, 1)


def _mapa_ruido():
    rng = np.random.default_rng(7)
    return np.round(rng.random((8, 8)), 2)


MAPAS = {
    "Activación (mancha)": _mapa_activacion,
    "Cuadrado": _mapa_cuadrado,
    "Diagonal": _mapa_diagonal,
    "Ruido": _mapa_ruido,
}


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def pooling(img, size=2, stride=2, modo="max"):
    """Pooling con ventana `size` y `stride`. modo: 'max' o 'avg'.

    Devuelve (salida, lista_de_celdas_ganadoras). Para 'avg' la lista va vacía;
    para 'max' contiene (fila, col) global del máximo de cada ventana.
    """
    H, W = img.shape
    oh = (H - size) // stride + 1
    ow = (W - size) // stride + 1
    out = np.zeros((oh, ow))
    ganadoras = []
    for r in range(oh):
        for c in range(ow):
            rr, cc = r * stride, c * stride
            ventana = img[rr:rr + size, cc:cc + size]
            if modo == "max":
                ai, aj = np.unravel_index(np.argmax(ventana), ventana.shape)
                out[r, c] = ventana[ai, aj]
                ganadoras.append((rr + ai, cc + aj))
            else:
                out[r, c] = ventana.mean()
    return out, ganadoras


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _grilla_base(ax, M, vmin=0, vmax=1, mascara=None):
    disp = np.ma.array(M, mask=mascara) if mascara is not None else M
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")
    ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    h, w = M.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#ddd", lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])


def _dibujar_entrada(ax, img, win, size, modo, ganadora):
    _grilla_base(ax, img)
    rr, cc = win
    ax.add_patch(mpatches.Rectangle((cc - 0.5, rr - 0.5), size, size, fill=False,
                 edgecolor="#e67e22", lw=3, zorder=5))
    if modo == "max" and ganadora is not None:
        gi, gj = ganadora
        ax.add_patch(mpatches.Rectangle((gj - 0.5, gi - 0.5), 1, 1, fill=False,
                     edgecolor="#c0392b", lw=3, zorder=6))
    ax.set_title(f"Mapa de entrada  ({img.shape[0]}×{img.shape[1]})",
                 fontsize=11, fontweight="bold")


def _dibujar_salida(ax, out, cur_rc, mascara):
    _grilla_base(ax, out, mascara=mascara)
    r, c = cur_rc
    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                 edgecolor="#c0392b", lw=3, zorder=5))
    ax.set_title(f"Salida ({out.shape[0]}×{out.shape[1]})",
                 fontsize=11, fontweight="bold")


def _minigrilla(ax, ox, oy, vals, ganadora_local, modo):
    cm = plt.get_cmap("viridis")
    n = vals.shape[0]
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            rgb = cm(v)[:3]
            es_win = (modo == "max" and ganadora_local == (i, j))
            ax.add_patch(mpatches.Rectangle((ox + j, oy - i), 1, 1, facecolor=rgb,
                         edgecolor="#c0392b" if es_win else "#888",
                         lw=2.6 if es_win else 0.8, zorder=4 if es_win else 3))
            txt = "white" if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]) < 0.5 else "#111"
            ax.text(ox + j + 0.5, oy - i + 0.5, f"{v:.2f}", ha="center",
                    va="center", fontsize=9, color=txt, zorder=5)


def figura_pooling(mapa="Activación (mancha)", modo="max", size=2, stride=2, paso=0):
    img = MAPAS[mapa]()
    out, ganadoras = pooling(img, size, stride, modo)
    oh, ow = out.shape
    n_out = oh * ow
    paso = int(np.clip(paso, 0, n_out - 1))
    r_o, c_o = divmod(paso, ow)
    rr, cc = r_o * stride, c_o * stride
    ventana = img[rr:rr + size, cc:cc + size]
    valor = out[r_o, c_o]

    ganadora = ganadoras[paso] if modo == "max" else None
    gan_local = (ganadora[0] - rr, ganadora[1] - cc) if ganadora else None

    idx = np.arange(n_out).reshape(oh, ow)
    mascara = idx > paso

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.3, 0.8])
    ax_in = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, :])

    _dibujar_entrada(ax_in, img, (rr, cc), size, modo, ganadora)
    _dibujar_salida(ax_out, out, (r_o, c_o), mascara)

    ax_comp.set_xlim(-0.5, 11.0)
    ax_comp.set_ylim(-2.4, size + 2.0)
    ax_comp.set_aspect("equal")
    ax_comp.axis("off")
    oy = size
    _minigrilla(ax_comp, 0, oy, ventana, gan_local, modo)
    ax_comp.text(size / 2, oy + 1.0, "ventana", ha="center", fontsize=9,
                 fontweight="bold")
    op = "máx" if modo == "max" else "promedio"
    ax_comp.text(size + 0.9, oy / 2 + 0.2, "→", fontsize=22, va="center", ha="center")
    bx, by, bw, bh = size + 1.7, oy / 2 - 0.55, 3.4, 1.6
    ax_comp.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
                      boxstyle="round,pad=0.06,rounding_size=0.12",
                      fc="#fdecea", ec="#c0392b", lw=2, zorder=3))
    ax_comp.text(bx + bw / 2, by + bh - 0.42, op, ha="center", va="center",
                 fontsize=11, color="#c0392b")
    ax_comp.text(bx + bw / 2, by + 0.55, f"{valor:.2f}", ha="center", va="center",
                 fontsize=19, fontweight="bold", color="#c0392b")

    nota = (f"El pooling NO tiene parámetros que aprender: es una operación fija "
            f"({op}) sobre cada ventana {size}×{size}.\n"
            f"Reduce el tamaño ({img.shape[0]}×{img.shape[1]} → {oh}×{ow}, "
            f"submuestreo) y da algo de invariancia a pequeñas traslaciones. "
            + ("Max conserva la activación más fuerte."
               if modo == "max" else "El promedio suaviza la región."))
    ax_comp.text(0, -1.5, nota, ha="left", va="center", fontsize=9.5, color="#333")

    fig.suptitle(f"{'Max' if modo=='max' else 'Average'} pooling  ·  «{mapa}»  ·  "
                 f"ventana {size}×{size}, stride {stride}  ·  posición {paso+1}/{n_out}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def pooling_interactiva(mapa="Activación (mancha)", modo="max"):
    """Despliega el pooling interactivo (max / average) sobre un mapa de ejemplo.

    Controles: mapa de entrada, tipo de pooling (max / average), tamaño de
    ventana, stride, una posición (slider para recorrer a mano) y un botón para
    animar el barrido. En max pooling se resalta en rojo la celda que gana.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    def n_posiciones():
        H = W = 8
        oh = (H - s_size.value) // s_str.value + 1
        ow = (W - s_size.value) // s_str.value + 1
        return oh * ow

    estado = {"paso": 0}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_pooling(s_map.value, s_modo.value,
                                 s_size.value, s_str.value, estado["paso"])
            plt.show()

    s_map = widgets.Dropdown(options=list(MAPAS.keys()), value=mapa,
                             description="mapa",
                             style={"description_width": "55px"},
                             layout=widgets.Layout(width="240px"))
    s_modo = widgets.ToggleButtons(options=[("Max", "max"), ("Average", "avg")],
                                   value=modo, description="tipo")
    s_size = widgets.IntSlider(value=2, min=2, max=3, step=1, description="ventana",
                               style={"description_width": "70px"},
                               layout=widgets.Layout(width="250px"))
    s_str = widgets.IntSlider(value=2, min=1, max=3, step=1, description="stride",
                              style={"description_width": "70px"},
                              layout=widgets.Layout(width="250px"))
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
    for w in (s_map, s_modo, s_size, s_str):
        w.observe(on_cfg, names="value")

    def on_run(_):
        for w in (b_run, s_map, s_modo, s_size, s_str, s_pos):
            w.disabled = True
        try:
            n = n_posiciones()
            for k in range(n):
                estado["paso"] = k
                redibujar()
                time.sleep(0.18)
            s_pos.unobserve(on_pos, names="value")
            s_pos.value = n - 1
            s_pos.observe(on_pos, names="value")
        finally:
            for w in (b_run, s_map, s_modo, s_size, s_str, s_pos):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Pooling: reducir el mapa</h3>"
        "<span style='color:#555'>Una ventana recorre el mapa y lo resume: "
        "<b>max</b> toma el valor más alto de cada región (en rojo el que gana), "
        "<b>average</b> toma el promedio. No hay pesos que aprender; el mapa "
        "queda más pequeño.</span>")
    controles = widgets.VBox([
        widgets.HBox([s_map, s_modo, b_run]),
        widgets.HBox([s_size, s_str]),
        s_pos,
    ])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
