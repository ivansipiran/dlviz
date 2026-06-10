"""Animación de un perceptrón multicapa (2-H-1, sigmoide) resolviendo XOR.

Arquitectura: 2 entradas -> 1 capa oculta de H neuronas -> 1 salida, todas con
activación sigmoide (dos capas de pesos). Pensado para Colab/Jupyter.

La función de alto nivel es :func:`mlp_xor_interactivo`. Anima el entrenamiento
por descenso de gradiente y muestra, en tiempo real, el diagrama de la red, la
frontera de decisión 2D y la curva de pérdida. La lógica de cómputo y de dibujo
se mantiene separada de la capa de widgets para poder probarla sin frontend.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Datos del problema XOR (fijos)
X_XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_XOR = np.array([[0.0], [1.0], [1.0], [0.0]])

_EPS = 1e-9


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -500, 500)))


# ---------------------------------------------------------------------------
# Red: estado, forward, backprop (numpy, full-batch)
# ---------------------------------------------------------------------------
def init_red(H=2, semilla=None):
    """Inicializa los parámetros de una red 2-H-1 con pesos aleatorios."""
    rng = np.random.default_rng(semilla)
    return {
        "W1": rng.normal(0, 1, (H, 2)),
        "b1": np.zeros(H),
        "W2": rng.normal(0, 1, (1, H)),
        "b2": np.zeros(1),
        # velocidades para momentum
        "vW1": np.zeros((H, 2)), "vb1": np.zeros(H),
        "vW2": np.zeros((1, H)), "vb2": np.zeros(1),
    }


def solucion_xor():
    """Pesos analíticos (H=2) que resuelven XOR: una neurona OR y una AND."""
    return {
        "W1": np.array([[20.0, 20.0], [20.0, 20.0]]),  # OR y AND
        "b1": np.array([-10.0, -30.0]),
        "W2": np.array([[20.0, -30.0]]),               # h1 AND (NOT h2)
        "b2": np.array([-10.0]),
        "vW1": np.zeros((2, 2)), "vb1": np.zeros(2),
        "vW2": np.zeros((1, 2)), "vb2": np.zeros(1),
    }


def forward(p, X):
    """Pasada forward. Devuelve (a1, a2) = (activaciones ocultas, salida)."""
    a1 = _sigmoid(X @ p["W1"].T + p["b1"])
    a2 = _sigmoid(a1 @ p["W2"].T + p["b2"])
    return a1, a2


def _bce(y, prob):
    prob = np.clip(prob, _EPS, 1 - _EPS)
    return float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean())


def paso_gd(p, lr=1.0, momentum=0.9):
    """Un paso de descenso de gradiente full-batch sobre XOR. Modifica p in-place."""
    X, Y, N = X_XOR, Y_XOR, len(X_XOR)
    a1, a2 = forward(p, X)

    dz2 = (a2 - Y) / N                      # BCE + sigmoide
    dW2 = dz2.T @ a1
    db2 = dz2.sum(0)
    dz1 = (dz2 @ p["W2"]) * a1 * (1 - a1)
    dW1 = dz1.T @ X
    db1 = dz1.sum(0)

    for key, grad in (("W2", dW2), ("b2", db2), ("W1", dW1), ("b1", db1)):
        v = "v" + key
        p[v] = momentum * p[v] - lr * grad
        p[key] = p[key] + p[v]

    return _bce(Y, a2)


def accuracy(p):
    _, a2 = forward(p, X_XOR)
    return float(((a2 > 0.5) == (Y_XOR > 0.5)).mean())


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _dibujar_red(ax, p):
    H = p["W1"].shape[0]
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")
    ax.set_title("Red multicapa  (2 → %d → 1)" % H, fontsize=12, fontweight="bold")

    def ys(n):
        return [0.5] if n == 1 else list(np.linspace(0.92, 0.08, n))

    in_pos = [(0.0, y) for y in ys(2)]
    hid_pos = [(1.3, y) for y in ys(H)]
    out_pos = [(2.3, y) for y in ys(1)]

    def arista(p0, p1, w):
        color = "#2c6fbb" if w >= 0 else "#c0392b"
        lw = 0.6 + 3.4 * min(abs(w), 8.0) / 8.0
        a = 0.35 + 0.55 * min(abs(w), 8.0) / 8.0
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, alpha=a, zorder=1)

    for j, hp in enumerate(hid_pos):
        for i, ip in enumerate(in_pos):
            arista(ip, hp, p["W1"][j, i])
    for k, op in enumerate(out_pos):
        for j, hp in enumerate(hid_pos):
            arista(hp, op, p["W2"][k, j])

    def nodo(pos, txt, color):
        ax.add_patch(mpatches.Circle(pos, 0.085, facecolor=color,
                                     edgecolor="#33425b", lw=1.4, zorder=3))
        ax.text(pos[0], pos[1], txt, ha="center", va="center", fontsize=9, zorder=4)

    for i, ip in enumerate(in_pos):
        nodo(ip, f"$x_{i+1}$", "#e8eef7")
    for hp in hid_pos:
        nodo(hp, "", "#dfe7f2")
    nodo(out_pos[0], r"$\hat{y}$", "#eef3ee")

    ax.text(0.0, 1.08, "entrada", ha="center", fontsize=9, color="#555")
    ax.text(1.3, 1.08, "oculta\n(sigmoide)", ha="center", fontsize=9, color="#555")
    ax.text(2.3, 1.08, "salida\n(sigmoide)", ha="center", fontsize=9, color="#555")
    ax.text(1.15, -0.16, "azul: peso +   rojo: peso −   (grosor ∝ |w|; sesgos también se entrenan)",
            ha="center", fontsize=7.5, color="#777")


def _dibujar_frontera(ax, p, lo=-0.5, hi=1.5):
    xs = np.linspace(lo, hi, 250)
    G1, G2 = np.meshgrid(xs, xs)
    grid = np.c_[G1.ravel(), G2.ravel()]
    _, prob = forward(p, grid)
    P = prob.reshape(G1.shape)

    ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 21), cmap="RdYlGn", alpha=0.9)
    ax.contour(G1, G2, P, levels=[0.5], colors="#222", linewidths=2.2)

    # Rectas de cada neurona oculta: W1[j]·x + b1[j] = 0
    H = p["W1"].shape[0]
    for j in range(H):
        w1, w2 = p["W1"][j]
        b = p["b1"][j]
        if abs(w2) > 1e-6:
            ax.plot(xs, -(w1 * xs + b) / w2, "--", color="#33425b",
                    lw=1.2, alpha=0.7, zorder=3)
        elif abs(w1) > 1e-6:
            ax.axvline(-b / w1, ls="--", color="#33425b", lw=1.2, alpha=0.7, zorder=3)

    # Los 4 puntos XOR: relleno = prob predicha, borde = clase verdadera
    _, py = forward(p, X_XOR)
    for (x1, x2), yt, pr in zip(X_XOR, Y_XOR.ravel(), py.ravel()):
        borde = "#1e8449" if yt == 1 else "#c0392b"
        relleno = plt.cm.RdYlGn(pr)
        ax.scatter([x1], [x2], s=320, facecolor=relleno, edgecolor=borde,
                   lw=3, zorder=5)
        ax.text(x1, x2, str(int(yt)), ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Frontera de decisión", fontsize=12, fontweight="bold")


def _dibujar_perdida(ax, hist):
    ax.set_title("Pérdida (BCE)", fontsize=11, fontweight="bold")
    if hist:
        ax.plot(hist, color="#2c6fbb", lw=1.8)
        ax.set_xlim(0, max(len(hist), 10))
        ax.set_ylim(0, max(0.05, max(hist) * 1.05))
    ax.set_xlabel("época")
    ax.grid(alpha=0.3)


def figura_mlp(p, hist=None, epoca=0, intento=None):
    """Construye la figura completa. Reutilizable y testeable sin widgets."""
    hist = hist or []
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[3, 1.1])
    ax_net = fig.add_subplot(gs[:, 0])
    ax_bnd = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])

    _dibujar_red(ax_net, p)
    _dibujar_frontera(ax_bnd, p)
    _dibujar_perdida(ax_loss, hist)

    acc = accuracy(p)
    loss = hist[-1] if hist else _bce(Y_XOR, forward(p, X_XOR)[1])
    estado = "✓ XOR resuelto" if acc == 1.0 else "entrenando…"
    prefijo = "" if intento is None else f"intento {intento}  ·  "
    fig.suptitle(f"{prefijo}época {epoca}  ·  pérdida {loss:.3f}  ·  "
                 f"exactitud {acc*100:.0f}%   {estado}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo con animación de entrenamiento
# ---------------------------------------------------------------------------
def mlp_xor_interactivo(H=2, lr=1.0, epocas=2500):
    """Despliega el MLP 2-H-1 que aprende XOR, con animación de entrenamiento.

    Controles: learning rate, número de neuronas ocultas y los botones
    Entrenar (anima el descenso de gradiente desde los pesos actuales),
    Reiniciar (pesos aleatorios nuevos) y Solución XOR (carga pesos analíticos
    que resuelven XOR con H=2). Si la red queda atascada en un mínimo local y no
    separa XOR, se avisa para que pulses Reiniciar y vuelvas a intentar.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    MOMENTUM = 0.9
    REDRAW = max(20, epocas // 60)      # ~60 fotogramas

    BANNER_OK = (
        "<div style='background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;"
        "padding:8px 12px;color:#1b5e20'>✓ <b>XOR resuelto.</b> La frontera de "
        "decisión dejó de ser lineal.</div>")
    BANNER_ATASCO = (
        "<div style='background:#fff3cd;border:1px solid #ffe08a;border-radius:6px;"
        "padding:8px 12px;color:#8a6d00'>⚠️ La red quedó <b>atascada en un mínimo "
        "local</b> y no separa XOR. Con pocas neuronas ocultas esto ocurre según "
        "la inicialización. Pulsa <b>↻ Reiniciar</b> para obtener pesos nuevos y "
        "vuelve a <b>Entrenar</b>.</div>")

    estado = {"p": init_red(H), "hist": [], "epoca": 0}

    out = widgets.Output()
    estado_html = widgets.HTML("")

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_mlp(estado["p"], estado["hist"], estado["epoca"])
            plt.show()

    s_lr = widgets.FloatLogSlider(value=lr, base=10, min=-1.3, max=0.7, step=0.05,
                                  description="learning rate",
                                  style={"description_width": "110px"},
                                  layout=widgets.Layout(width="320px"),
                                  readout_format=".2f")
    s_H = widgets.IntSlider(value=H, min=2, max=6, step=1,
                            description="neuronas ocultas",
                            style={"description_width": "110px"},
                            layout=widgets.Layout(width="320px"))
    b_train = widgets.Button(description="▶ Entrenar", button_style="success")
    b_reset = widgets.Button(description="↻ Reiniciar")
    b_sol = widgets.Button(description="Solución XOR")

    def set_disabled(v):
        for w in (b_train, b_reset, b_sol, s_lr, s_H):
            w.disabled = v

    def on_H(change):
        estado["p"] = init_red(s_H.value)
        estado["hist"] = []
        estado["epoca"] = 0
        estado_html.value = ""
        redibujar()
    s_H.observe(on_H, names="value")

    def on_reset(_):
        estado["p"] = init_red(s_H.value)
        estado["hist"] = []
        estado["epoca"] = 0
        estado_html.value = ""
        redibujar()
    b_reset.on_click(on_reset)

    def on_sol(_):
        s_H.unobserve(on_H, names="value")
        s_H.value = 2
        s_H.observe(on_H, names="value")
        estado["p"] = solucion_xor()
        estado["hist"] = []
        estado["epoca"] = 0
        estado_html.value = BANNER_OK
        redibujar()
    b_sol.on_click(on_sol)

    def on_train(_):
        set_disabled(True)
        estado_html.value = ""
        try:
            lr_v = s_lr.value
            estado["hist"] = []          # curva de pérdida nueva para esta corrida
            estado["epoca"] = 0
            for e in range(epocas):
                loss = paso_gd(estado["p"], lr=lr_v, momentum=MOMENTUM)
                estado["hist"].append(loss)
                estado["epoca"] = e
                if e % REDRAW == 0 or e == epocas - 1:
                    redibujar()
                    time.sleep(0.01)
            estado_html.value = (BANNER_OK if accuracy(estado["p"]) == 1.0
                                 else BANNER_ATASCO)
        finally:
            set_disabled(False)
    b_train.on_click(on_train)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Perceptrón multicapa resolviendo XOR</h3>"
        "<span style='color:#555'>Una sola neurona no puede separar XOR; con una "
        "capa oculta sí. Pulsa <b>Entrenar</b> y observa cómo la frontera de "
        "decisión deja de ser lineal.</span>")
    controles = widgets.VBox([
        widgets.HBox([s_lr, s_H]),
        widgets.HBox([b_train, b_reset, b_sol]),
    ])

    redibujar()
    display(widgets.VBox([titulo, controles, estado_html, out]))
