"""Carrera animada de optimizadores de Deep Learning.

Compara SGD, Momentum, RMSProp y Adam sobre superficies de pérdida 2D, mostrando
sus trayectorias sobre las curvas de nivel, las curvas de convergencia en escala
logarítmica y un marcador en vivo de cuánto ha convergido cada uno. Pensado para
Colab/Jupyter.

Función de alto nivel: :func:`optimizadores_interactivo`. La lógica de cómputo
(superficies y optimizadores) y de dibujo se mantiene separada de los widgets
para poder probarla sin frontend.

Para añadir una superficie nueva basta con agregar una entrada a SUPERFICIES con
su función, gradiente, dominio, mínimo, punto de partida, número de pasos y un
learning rate base por optimizador.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Superficies de pérdida
# ---------------------------------------------------------------------------
def _rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


SUPERFICIES = {
    "Valle mal condicionado": {
        "f": lambda p: p[0]**2 + 100 * p[1]**2,
        "grad": lambda p: np.array([2 * p[0], 200 * p[1]]),
        "xlim": (-1.2, 1.2), "ylim": (-1.2, 1.2),
        "min": (0.0, 0.0), "start": (-1.0, 1.0), "pasos": 60,
        "levels": [0.1, 1, 2, 4, 9, 16, 25, 36, 49, 64, 81, 100],
        "lr": {"SGD": 0.008, "Momentum": 0.002, "RMSProp": 0.03, "Adam": 0.08},
        "desc": r"$f(x,y)=x^2+100\,y^2$  ·  mínimo en $(0,0)$",
    },
    "Rosenbrock (valle curvo)": {
        "f": lambda p: (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2,
        "grad": _rosenbrock_grad,
        "xlim": (-2.0, 2.0), "ylim": (-1.0, 3.0),
        "min": (1.0, 1.0), "start": (-1.3, 1.3), "pasos": 600,
        "levels": "log",
        "lr": {"SGD": 0.002, "Momentum": 0.0006, "RMSProp": 0.05, "Adam": 0.2},
        "desc": r"$f(x,y)=(1-x)^2+100\,(y-x^2)^2$  ·  mínimo en $(1,1)$",
    },
}

ORDEN_OPT = ["SGD", "Momentum", "RMSProp", "Adam"]
COLOR_OPT = {"SGD": "#7f8c8d", "Momentum": "#2c6fbb",
             "RMSProp": "#1e8449", "Adam": "#d35400"}


# ---------------------------------------------------------------------------
# Optimizadores
# ---------------------------------------------------------------------------
def optimizar(surf, opt, x0, lr, beta=0.9, beta2=0.999, eps=1e-8, pasos=60):
    """Corre un optimizador y devuelve (trayectoria (P+1,2), pérdidas (P+1,))."""
    f, grad = surf["f"], surf["grad"]
    lo = np.array([surf["xlim"][0] - 0.4, surf["ylim"][0] - 0.4])
    hi = np.array([surf["xlim"][1] + 0.4, surf["ylim"][1] + 0.4])

    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    m = np.zeros(2)   # 1er momento (Adam) / no usado en otros
    s = np.zeros(2)   # 2do momento (RMSProp/Adam)
    v = np.zeros(2)   # velocidad (Momentum)

    for t in range(1, pasos + 1):
        g = grad(x)
        if not np.all(np.isfinite(g)):
            break
        g = np.clip(g, -1e6, 1e6)

        if opt == "SGD":
            x = x - lr * g
        elif opt == "Momentum":
            v = beta * v + g
            x = x - lr * v
        elif opt == "RMSProp":
            s = beta * s + (1 - beta) * g**2
            x = x - lr * g / (np.sqrt(s) + eps)
        elif opt == "Adam":
            m = beta * m + (1 - beta) * g
            s = beta2 * s + (1 - beta2) * g**2
            mc = m / (1 - beta**t)
            sc = s / (1 - beta2**t)
            x = x - lr * mc / (np.sqrt(sc) + eps)
        else:
            raise ValueError(opt)

        if not np.all(np.isfinite(x)):
            break
        x = np.clip(x, lo, hi)
        traj.append(x.copy())

    traj = np.array(traj)
    losses = np.array([f(p) for p in traj])
    return traj, losses


def simular(surf_key, x0, lr_mult, beta, opts):
    """Corre todos los optimizadores seleccionados. Devuelve un dict de resultados."""
    surf = SUPERFICIES[surf_key]
    pasos = surf["pasos"]
    trajs, losses = {}, {}
    for opt in opts:
        lr = surf["lr"][opt] * lr_mult
        tr, ls = optimizar(surf, opt, x0, lr, beta=beta, pasos=pasos)
        trajs[opt] = tr
        losses[opt] = ls

    # ~60 fotogramas, subsampleando si hay muchos pasos
    n_frames = min(60, pasos)
    frames = np.unique(np.linspace(0, pasos, n_frames + 1).astype(int))
    return {"trajs": trajs, "losses": losses, "frames": frames,
            "opts": list(opts), "pasos": pasos}


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _grid_Z(surf):
    xs = np.linspace(*surf["xlim"], 160)
    ys = np.linspace(*surf["ylim"], 160)
    GX, GY = np.meshgrid(xs, ys)
    Z = surf["f"](np.array([GX, GY]))
    return GX, GY, Z


def _dibujar_superficie(ax, surf_key, sim, k, start):
    surf = SUPERFICIES[surf_key]
    GX, GY, Z = _grid_Z(surf)

    if surf["levels"] == "log":
        lo = max(Z.min(), 1e-2)
        levels = np.logspace(np.log10(lo), np.log10(Z.max()), 18)
    else:
        levels = surf["levels"]
    ax.contour(GX, GY, Z, levels=levels, cmap="Blues_r", linewidths=0.8, alpha=0.8)

    mx, my = surf["min"]
    ax.plot([mx], [my], marker="*", ms=20, color="#f1c40f",
            markeredgecolor="#7d6608", zorder=6)

    if sim and sim["trajs"]:
        for opt in sim["opts"]:
            tr = sim["trajs"][opt]
            kk = min(k, len(tr) - 1)
            c = COLOR_OPT[opt]
            ax.plot(tr[:kk + 1, 0], tr[:kk + 1, 1], "-", color=c, lw=2,
                    alpha=0.9, zorder=4, label=opt)
            ax.plot([tr[kk, 0]], [tr[kk, 1]], "o", color=c, ms=9,
                    markeredgecolor="white", markeredgewidth=1.3, zorder=5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        ax.plot([start[0]], [start[1]], "o", color="#333", ms=9, zorder=5)
        ax.annotate("inicio", start, textcoords="offset points", xytext=(8, 8),
                    fontsize=9)

    ax.set_xlim(*surf["xlim"])
    ax.set_ylim(*surf["ylim"])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Trayectorias sobre la superficie de pérdida",
                 fontsize=12, fontweight="bold")


def _dibujar_convergencia(ax, sim, k):
    ax.set_title("Convergencia (pérdida, escala log)", fontsize=11, fontweight="bold")
    if sim and sim["trajs"]:
        for opt in sim["opts"]:
            ls = np.clip(sim["losses"][opt], 1e-12, None)
            kk = min(k, len(ls) - 1)
            ax.semilogy(range(kk + 1), ls[:kk + 1], color=COLOR_OPT[opt], lw=2)
        ax.set_xlim(0, sim["pasos"])
    ax.set_xlabel("iteración")
    ax.set_ylabel("pérdida")
    ax.grid(alpha=0.3, which="both")


def _dibujar_marcador(ax, sim, k):
    ax.set_title("Órdenes de magnitud reducidos", fontsize=11, fontweight="bold")
    ax.axis("off")
    if not (sim and sim["trajs"]):
        return
    datos = []
    for opt in sim["opts"]:
        ls = sim["losses"][opt]
        kk = min(k, len(ls) - 1)
        l0 = max(ls[0], 1e-12)
        lk = max(ls[kk], 1e-12)
        reducido = max(0.0, np.log10(l0) - np.log10(lk))
        datos.append((opt, reducido, ls[kk]))
    datos.sort(key=lambda d: d[1], reverse=True)

    maxr = max((d[1] for d in datos), default=1.0) or 1.0
    for i, (opt, red, loss) in enumerate(datos):
        y = len(datos) - 1 - i
        ax.barh(y, red, height=0.6, color=COLOR_OPT[opt], alpha=0.9)
        ax.text(0, y + 0.42, f"{opt}", fontsize=9, fontweight="bold",
                color=COLOR_OPT[opt])
        ax.text(maxr * 1.02, y, f"pérdida {loss:.1e}", va="center", fontsize=8,
                color="#444")
    ax.set_xlim(0, maxr * 1.45)
    ax.set_ylim(-0.6, len(datos) - 0.2)


def figura_optim(surf_key, sim, k, start):
    """Construye la figura completa (superficie + convergencia + marcador)."""
    fig = plt.figure(figsize=(12.5, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.3, 1])
    ax_s = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    _dibujar_superficie(ax_s, surf_key, sim, k, start)
    _dibujar_convergencia(ax_c, sim, k)
    _dibujar_marcador(ax_m, sim, k)

    it = 0 if not sim else min(k, sim["pasos"])
    fig.suptitle(f"{SUPERFICIES[surf_key]['desc']}      ·      iteración {it}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def optimizadores_interactivo(superficie="Valle mal condicionado"):
    """Anima una carrera de optimizadores (SGD, Momentum, RMSProp, Adam).

    Controles: superficie de pérdida, qué optimizadores incluir, punto de
    partida, un multiplicador de learning rate (cada optimizador parte de un lr
    base sensato por superficie) y beta. El botón Animar corre el descenso paso
    a paso mostrando las trayectorias, las curvas de convergencia en escala log
    y cuántos órdenes de magnitud de pérdida ha reducido cada optimizador.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"sim": None, "start": SUPERFICIES[superficie]["start"]}
    out = widgets.Output()

    def redibujar(k):
        with out:
            out.clear_output(wait=True)
            fig = figura_optim(s_surf.value, estado["sim"], k, estado["start"])
            plt.show()

    s_surf = widgets.Dropdown(options=list(SUPERFICIES.keys()), value=superficie,
                              description="superficie",
                              style={"description_width": "90px"},
                              layout=widgets.Layout(width="360px"))
    cbs = {opt: widgets.Checkbox(value=True, description=opt, indent=False,
                                 layout=widgets.Layout(width="120px"))
           for opt in ORDEN_OPT}
    s_lr = widgets.FloatLogSlider(value=1.0, base=10, min=-0.6, max=0.6, step=0.05,
                                  description="learning rate ×",
                                  style={"description_width": "110px"},
                                  layout=widgets.Layout(width="330px"),
                                  readout_format=".2f")
    s_beta = widgets.FloatSlider(value=0.9, min=0.5, max=0.99, step=0.01,
                                 description="beta",
                                 style={"description_width": "110px"},
                                 layout=widgets.Layout(width="330px"),
                                 readout_format=".2f")

    def _slider_xy(eje):
        surf = SUPERFICIES[superficie]
        lim = surf["xlim"] if eje == 0 else surf["ylim"]
        return widgets.FloatSlider(value=surf["start"][eje], min=lim[0], max=lim[1],
                                   step=0.05, description=f"inicio {'x₁' if eje==0 else 'x₂'}",
                                   style={"description_width": "70px"},
                                   layout=widgets.Layout(width="280px"),
                                   readout_format=".2f")
    s_x0 = _slider_xy(0)
    s_y0 = _slider_xy(1)

    b_animar = widgets.Button(description="▶ Animar", button_style="success")
    b_reset = widgets.Button(description="↺ Limpiar")

    def set_disabled(v):
        for w in (b_animar, b_reset, s_surf, s_lr, s_beta, s_x0, s_y0, *cbs.values()):
            w.disabled = v

    def on_surf(_):
        surf = SUPERFICIES[s_surf.value]
        for sld, eje in ((s_x0, 0), (s_y0, 1)):
            lim = surf["xlim"] if eje == 0 else surf["ylim"]
            sld.unobserve(on_start, names="value")
            sld.min, sld.max = lim[0], lim[1]
            sld.value = surf["start"][eje]
            sld.observe(on_start, names="value")
        estado["sim"] = None
        estado["start"] = surf["start"]
        redibujar(0)
    s_surf.observe(on_surf, names="value")

    def on_start(_):
        estado["start"] = (s_x0.value, s_y0.value)
        estado["sim"] = None
        redibujar(0)
    s_x0.observe(on_start, names="value")
    s_y0.observe(on_start, names="value")

    def on_reset(_):
        estado["sim"] = None
        redibujar(0)
    b_reset.on_click(on_reset)

    def on_animar(_):
        opts = [opt for opt in ORDEN_OPT if cbs[opt].value]
        if not opts:
            return
        set_disabled(True)
        try:
            estado["start"] = (s_x0.value, s_y0.value)
            estado["sim"] = simular(s_surf.value, estado["start"],
                                    s_lr.value, s_beta.value, opts)
            for k in estado["sim"]["frames"]:
                redibujar(int(k))
                time.sleep(0.03)
        finally:
            set_disabled(False)
    b_animar.on_click(on_animar)

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Carrera de optimizadores</h3>"
        "<span style='color:#555'>Cada optimizador baja por la superficie de "
        "pérdida desde el mismo punto. Observa quién llega antes al mínimo (★), "
        "cómo SGD zigzaguea en el valle y cómo los métodos adaptativos lo "
        "evitan.</span>")
    controles = widgets.VBox([
        widgets.HBox([s_surf, widgets.HBox(list(cbs.values()))]),
        widgets.HBox([s_lr, s_beta]),
        widgets.HBox([s_x0, s_y0]),
        widgets.HBox([b_animar, b_reset]),
    ])

    redibujar(0)
    display(widgets.VBox([titulo, controles, out]))
