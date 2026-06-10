"""Visualización interactiva de un perceptrón simple (2 entradas, 1 salida).

Pensado para ejecutarse en Google Colab o Jupyter. Expone una única función
de alto nivel, :func:`perceptron_interactivo`, que despliega los controles y
los gráficos. La lógica de dibujo se mantiene en funciones independientes para
poder probarla sin un frontend de widgets.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Funciones de activación
# ---------------------------------------------------------------------------
def _step(z):
    """Función escalón (thresholding): 1 si z >= 0, 0 en caso contrario."""
    return np.where(np.asarray(z, dtype=float) >= 0.0, 1.0, 0.0)


def _sigmoid(z):
    """Función sigmoide: 1 / (1 + e^-z)."""
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


ACTIVACIONES = {
    "Step (umbral)": _step,
    "Sigmoide": _sigmoid,
}


def _forward(x1, x2, w1, w2, b, act):
    """Calcula z y la salida ŷ del perceptrón para una entrada escalar."""
    z = w1 * x1 + w2 * x2 + b
    y = float(act(np.array(z)))
    return z, y


# ---------------------------------------------------------------------------
# Dibujo del diagrama del perceptrón
# ---------------------------------------------------------------------------
def _dibujar_perceptron(ax, x1, x2, w1, w2, b, z, y, nombre_act):
    ax.set_xlim(-0.6, 7.0)
    ax.set_ylim(-2.6, 2.6)
    ax.axis("off")
    ax.set_title("Estructura del perceptrón", fontsize=13, fontweight="bold")

    # Posiciones de los nodos
    n_x1 = (0.5, 1.35)
    n_x2 = (0.5, -1.35)
    n_bias = (0.5, 0.0)
    n_sum = (3.0, 0.0)
    box_act = (4.6, 0.0)   # centro de la caja de activación
    n_out = (6.4, 0.0)

    def nodo(centro, texto, color="#e8eef7", r=0.46, fs=11):
        ax.add_patch(mpatches.Circle(centro, r, facecolor=color,
                                     edgecolor="#33425b", lw=1.6, zorder=3))
        ax.text(centro[0], centro[1], texto, ha="center", va="center",
                fontsize=fs, zorder=4)

    def arista(p0, p1, w=None, etiqueta=True, dy=0.22):
        if w is None:
            color, lw = "#33425b", 1.8
        else:
            color = "#2c6fbb" if w >= 0 else "#c0392b"
            lw = 1.0 + 4.0 * min(abs(w), 3.0) / 3.0
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=20, shrinkB=20), zorder=2)
        if etiqueta and w is not None:
            mx = p0[0] + 0.62 * (p1[0] - p0[0])
            my = p0[1] + 0.62 * (p1[1] - p0[1])
            ax.text(mx, my + dy, f"{w:+.2f}", color=color, fontsize=10,
                    ha="center", fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # Aristas entrada -> suma (peso codificado en color y grosor)
    arista(n_x1, n_sum, w1, dy=0.25)
    arista(n_x2, n_sum, w2, dy=-0.25)
    arista(n_bias, n_sum, b, dy=0.22)

    # Nodos de entrada y sesgo
    nodo(n_x1, f"$x_1$\n{x1:.1f}")
    nodo(n_x2, f"$x_2$\n{x2:.1f}")
    nodo(n_bias, "1", color="#f3ead8", r=0.34, fs=11)
    nodo(n_sum, r"$\Sigma$", color="#dfe7f2", r=0.52, fs=16)

    # Caja de activación
    w_box, h_box = 1.15, 1.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (box_act[0] - w_box / 2, box_act[1] - h_box / 2), w_box, h_box,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#eef3ee", edgecolor="#33425b", lw=1.6, zorder=3))
    simbolo = r"$\sigma$" if nombre_act.startswith("Sig") else r"$\theta$"
    ax.text(box_act[0], box_act[1] + 0.16, simbolo, ha="center", va="center",
            fontsize=15, zorder=4)
    ax.text(box_act[0], box_act[1] - 0.28, nombre_act.split(" ")[0],
            ha="center", va="center", fontsize=8, color="#555", zorder=4)

    # Suma -> activación -> salida
    arista(n_sum, (box_act[0] - w_box / 2, 0.0), etiqueta=False)
    arista((box_act[0] + w_box / 2, 0.0), n_out, etiqueta=False)

    # Nodo de salida coloreado según la clase predicha
    col_out = "#1e8449" if y >= 0.5 else "#c0392b"
    nodo(n_out, f"$\\hat{{y}}$\n{y:.2f}", color=col_out, fs=11)
    ax.text(n_out[0], n_out[1] - 0.78, "clase 1" if y >= 0.5 else "clase 0",
            ha="center", fontsize=9, color=col_out, fontweight="bold")

    # Lectura de la pasada forward
    txt = (rf"$z = w_1 x_1 + w_2 x_2 + b = "
           rf"({w1:.2f})({x1:.1f}) + ({w2:.2f})({x2:.1f}) + ({b:.2f}) = {z:.2f}$")
    ax.text(0.5, -2.35, txt, ha="left", va="center", fontsize=10,
            transform=ax.transData, color="#222")


# ---------------------------------------------------------------------------
# Dibujo de la frontera de decisión
# ---------------------------------------------------------------------------
def _dibujar_frontera(ax, x1, x2, w1, w2, b, act, nombre_act, fig=None, rango=3.0):
    xs = np.linspace(-rango, rango, 300)
    X1, X2 = np.meshgrid(xs, xs)
    Z = w1 * X1 + w2 * X2 + b
    A = act(Z)

    es_step = nombre_act.startswith("Step")
    if es_step:
        ax.contourf(X1, X2, A, levels=[-0.5, 0.5, 1.5],
                    colors=["#fdecea", "#e8f3ec"], alpha=0.95)
    else:
        cf = ax.contourf(X1, X2, A, levels=np.linspace(0, 1, 21),
                         cmap="RdYlGn", alpha=0.9)
        if fig is not None:
            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("ŷ (probabilidad)", fontsize=9)

    # Frontera de decisión: z = 0
    ax.contour(X1, X2, Z, levels=[0], colors="#33425b", linewidths=2.2)

    # Punto de entrada actual
    _, y = _forward(x1, x2, w1, w2, b, act)
    col = "#1e8449" if y >= 0.5 else "#c0392b"
    ax.scatter([x1], [x2], s=220, color=col, edgecolor="white",
               lw=2.2, zorder=6)
    ax.annotate(f"({x1:.1f}, {x2:.1f})", (x1, x2), textcoords="offset points",
                xytext=(12, 10), fontsize=10, fontweight="bold", color="#222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.85))

    ax.set_xlim(-rango, rango)
    ax.set_ylim(-rango, rango)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Frontera de decisión", fontsize=13, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)


def _figura(x1, x2, w1, w2, b, nombre_act, rango=3.0):
    """Construye la figura completa (diagrama + frontera). Reutilizable en tests."""
    act = ACTIVACIONES[nombre_act]
    z, y = _forward(x1, x2, w1, w2, b, act)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    _dibujar_perceptron(ax1, x1, x2, w1, w2, b, z, y, nombre_act)
    _dibujar_frontera(ax2, x1, x2, w1, w2, b, act, nombre_act, fig=fig, rango=rango)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo (requiere ipywidgets + Colab/Jupyter)
# ---------------------------------------------------------------------------
def perceptron_interactivo(rango: float = 3.0):
    """Despliega un perceptrón interactivo de 2 entradas y 1 salida.

    Permite modificar entradas (x1, x2), pesos (w1, w2), sesgo (b) y la función
    de activación (escalón o sigmoide). A la izquierda se muestra el diagrama
    del perceptrón con la pasada forward; a la derecha, la frontera de decisión
    en el plano (x1, x2) con el punto de entrada actual.

    Parameters
    ----------
    rango : float
        Límite del rango de los ejes y de los sliders (de -rango a +rango).
    """
    import ipywidgets as widgets
    from IPython.display import display

    estilo = {"description_width": "40px"}
    layout = widgets.Layout(width="240px")

    def slider(val, desc):
        return widgets.FloatSlider(value=val, min=-rango, max=rango, step=0.1,
                                   description=desc, style=estilo, layout=layout,
                                   continuous_update=True, readout_format=".1f")

    sx1 = slider(1.0, "$x_1$")
    sx2 = slider(0.5, "$x_2$")
    sw1 = slider(1.0, "$w_1$")
    sw2 = slider(-1.0, "$w_2$")
    sb = slider(0.0, "$b$")
    act = widgets.Dropdown(options=list(ACTIVACIONES.keys()),
                           value="Step (umbral)", description="Activación",
                           style={"description_width": "70px"},
                           layout=widgets.Layout(width="240px"))

    out = widgets.Output()

    def actualizar(_=None):
        with out:
            out.clear_output(wait=True)
            fig = _figura(sx1.value, sx2.value, sw1.value, sw2.value,
                          sb.value, act.value, rango=rango)
            plt.show()

    for w in (sx1, sx2, sw1, sw2, sb, act):
        w.observe(actualizar, names="value")

    col_entradas = widgets.VBox(
        [widgets.HTML("<b>Entradas</b>"), sx1, sx2])
    col_pesos = widgets.VBox(
        [widgets.HTML("<b>Pesos y sesgo</b>"), sw1, sw2, sb])
    col_act = widgets.VBox(
        [widgets.HTML("<b>Función de activación</b>"), act])
    controles = widgets.HBox([col_entradas, col_pesos, col_act])

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Perceptrón interactivo</h3>"
        "<span style='color:#555'>Mueve los sliders para ver la pasada "
        "forward y cómo cambia la frontera de decisión.</span>")

    actualizar()
    display(widgets.VBox([titulo, controles, out]))
