"""Animación didáctica de la atención (self-attention) sobre una secuencia.

Primer contacto con el mecanismo de atención: una secuencia de entrada, la
operación de atención (similitud → softmax → promedio ponderado) y la secuencia
de salida. Pensado para Colab/Jupyter.

Idea visual para distinguir los ingredientes:
  * Los *queries* y *keys* son embeddings semánticos de cada token: deciden
    CUÁNTA atención recibe cada token (los pesos).
  * Los *values* son colores: es lo que se MEZCLA. Así la salida de cada token
    es literalmente una mezcla de colores ponderada por la atención.
En un Transformer real, Q, K y V son proyecciones lineales aprendidas de la
entrada (Q=XWq, K=XWk, V=XWv); aquí se simplifican para ver el mecanismo.

Función de alto nivel: :func:`atencion_interactiva`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Frase de ejemplo y embeddings semánticos (queries = keys).
# Dimensiones interpretables: [DET, SUJETO, ACCIÓN, OBJETO, FRÍO]
TOKENS = ["El", "gato", "bebe", "leche", "fría"]
_EMB = np.array([
    [1.0, 0.7, 0.0, 0.0, 0.0],   # El    -> determinante del sujeto
    [0.3, 1.0, 0.5, 0.0, 0.0],   # gato  -> sujeto
    [0.0, 0.6, 1.0, 0.6, 0.0],   # bebe  -> acción (une sujeto y objeto)
    [0.0, 0.0, 0.5, 1.0, 0.5],   # leche -> objeto
    [0.0, 0.0, 0.0, 0.6, 1.0],   # fría  -> adjetivo del objeto
])
_EMB = _EMB / np.linalg.norm(_EMB, axis=1, keepdims=True)

# Values = colores (lo que se mezcla). Tonos bien separados para ver la mezcla.
COLORES = np.array([
    [0.557, 0.267, 0.678],   # El    morado
    [0.902, 0.494, 0.133],   # gato  naranjo
    [0.086, 0.627, 0.522],   # bebe  verde azulado
    [0.161, 0.502, 0.725],   # leche azul
    [0.753, 0.224, 0.169],   # fría  rojo
])


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def pesos_atencion(nitidez=4.0):
    """Matriz de pesos de atención NxN (cada fila suma 1)."""
    return _softmax((_EMB @ _EMB.T) * nitidez)


def colores_salida(W):
    """Color de cada token de salida = promedio ponderado de los values (colores)."""
    return np.clip(W @ COLORES, 0, 1)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _chip(ax, x, y, texto, facecolor, edgecolor="#33425b", lw=1.4,
          w=0.95, h=0.5, fontcolor="white", fontsize=12):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", color=fontcolor,
            fontsize=fontsize, fontweight="bold", zorder=4)


def _dibujar_flujo(ax, q, W):
    n = len(TOKENS)
    out_cols = colores_salida(W)
    xs = np.arange(n) * 1.5
    y_in, y_out = 1.6, 0.0
    ax.set_xlim(-1.0, xs[-1] + 1.0)
    ax.set_ylim(-0.9, 2.5)
    ax.axis("off")

    # Líneas de atención del query q hacia cada token de entrada
    for i in range(n):
        peso = W[q, i]
        ax.plot([xs[i], xs[q]], [y_in - 0.25, y_out + 0.25],
                color=tuple(COLORES[i]), lw=0.8 + 9 * peso,
                alpha=0.18 + 0.72 * peso, solid_capstyle="round", zorder=1)

    # Secuencia de entrada (values = color)
    for i, t in enumerate(TOKENS):
        _chip(ax, xs[i], y_in, t, tuple(COLORES[i]))
    # Secuencia de salida (mezcla); resaltar el token consultado
    for i, t in enumerate(TOKENS):
        seleccion = (i == q)
        _chip(ax, xs[i], y_out, t, tuple(out_cols[i]),
              edgecolor="#111" if seleccion else "#9aa0a6",
              lw=3.2 if seleccion else 1.0,
              fontcolor="white")

    ax.text(-0.95, y_in, "entrada", ha="right", va="center", fontsize=10,
            color="#555", fontweight="bold")
    ax.text(-0.95, y_out, "salida", ha="right", va="center", fontsize=10,
            color="#555", fontweight="bold")
    ax.text(xs[q], y_out - 0.55, f"«{TOKENS[q]}» mira a la secuencia\n"
            "y se vuelve una mezcla ponderada de los colores",
            ha="center", va="top", fontsize=9, color="#333")
    ax.set_title("Secuencia de entrada  →  atención  →  secuencia de salida",
                 fontsize=12, fontweight="bold")


def _dibujar_matriz(ax, W, q):
    n = len(TOKENS)
    ax.imshow(W, cmap="Purples", vmin=0, vmax=1, aspect="equal")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#222" if W[i, j] < 0.6 else "white")
    ax.set_xticks(range(n)); ax.set_xticklabels(TOKENS, fontsize=8, rotation=30)
    ax.set_yticks(range(n)); ax.set_yticklabels(TOKENS, fontsize=8)
    ax.set_xlabel("key (a quién mira)", fontsize=9)
    ax.set_ylabel("query (quién mira)", fontsize=9)
    ax.add_patch(mpatches.Rectangle((-0.5, q - 0.5), n, 1, fill=False,
                                    edgecolor="#c0392b", lw=2.5, zorder=5))
    ax.set_title("Matriz de atención (cada fila suma 1)", fontsize=11,
                 fontweight="bold")


def _dibujar_pesos(ax, W, q):
    n = len(TOKENS)
    w = W[q]
    y = np.arange(n)[::-1]
    ax.barh(y, w, color=[tuple(c) for c in COLORES], height=0.6)
    for yi, ti, wi in zip(y, TOKENS, w):
        ax.text(0.01, yi, ti, va="center", ha="left", fontsize=9,
                fontweight="bold", color="white" if wi > 0.12 else "#333")
        ax.text(wi + 0.01, yi, f"{wi:.2f}", va="center", fontsize=8, color="#444")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("peso de atención")
    ax.set_title(f"¿A qué atiende «{TOKENS[q]}»?", fontsize=11, fontweight="bold")


def figura_atencion(q=2, nitidez=4.0):
    """Figura completa: flujo entrada→salida, matriz de atención y pesos del query."""
    W = pesos_atencion(nitidez)
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax_flujo = fig.add_subplot(gs[:, 0])
    ax_mat = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])

    _dibujar_flujo(ax_flujo, q, W)
    _dibujar_matriz(ax_mat, W, q)
    _dibujar_pesos(ax_bar, W, q)

    fig.suptitle(r"Atención: $\mathrm{softmax}\!\left(QK^\top/\sqrt{d_k}\right)V$"
                 "      ·      Q,K = significado (deciden el peso)   ·   "
                 "V = color (lo que se mezcla)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def atencion_interactiva():
    """Despliega la animación interactiva de atención sobre la frase de ejemplo.

    Controles: el token que consulta (query), la nitidez del softmax (el papel
    de la escala 1/√dₖ: baja = atención repartida, alta = concentrada) y un
    botón para recorrer la secuencia token por token.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"q": 2}
    out = widgets.Output()

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_atencion(estado["q"], s_nit.value)
            plt.show()

    s_q = widgets.Dropdown(options=[(t, i) for i, t in enumerate(TOKENS)],
                           value=2, description="query",
                           style={"description_width": "70px"},
                           layout=widgets.Layout(width="240px"))
    s_nit = widgets.FloatSlider(value=4.0, min=0.5, max=9.0, step=0.5,
                                description="nitidez (escala softmax)",
                                style={"description_width": "160px"},
                                layout=widgets.Layout(width="380px"),
                                readout_format=".1f")
    b_run = widgets.Button(description="▶ Recorrer secuencia", button_style="success")

    def on_q(_):
        estado["q"] = s_q.value
        redibujar()
    s_q.observe(on_q, names="value")
    s_nit.observe(lambda _: redibujar(), names="value")

    def on_run(_):
        for w in (b_run, s_q, s_nit):
            w.disabled = True
        try:
            for i in range(len(TOKENS)):
                estado["q"] = i
                s_q.unobserve(on_q, names="value")
                s_q.value = i
                s_q.observe(on_q, names="value")
                redibujar()
                time.sleep(0.9)
        finally:
            for w in (b_run, s_q, s_nit):
                w.disabled = False
    b_run.on_click(on_run)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>¿Qué es la atención?</h3>"
        "<span style='color:#555'>Cada token de la secuencia <b>mira</b> a los "
        "demás y decide a cuáles prestar atención por <b>similitud</b> "
        "(query·key). El <b>softmax</b> convierte esas similitudes en pesos que "
        "suman 1, y la salida de cada token es el <b>promedio ponderado</b> de "
        "los <i>values</i> (aquí, colores). Elige el token que consulta y "
        "observa a qué atiende y de qué color resulta.</span>")
    controles = widgets.VBox([widgets.HBox([s_q, b_run]), s_nit])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
