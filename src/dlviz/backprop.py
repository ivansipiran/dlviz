"""Animación paso a paso de backpropagation en un MLP pequeño (2-2-2, sigmoide).

Recorre una iteración completa de entrenamiento sobre UN ejemplo (x, y):

  1. FORWARD      : cada neurona calcula z = w·a + b y su activación a = σ(z).
  2. ERROR/COSTO  : e_k = a_k - y_k  y  C = ½ Σ_k (a_k - y_k)².
  3. BACKWARD     : δ de cada neurona (regla de la cadena), empezando por la
                    salida y retropropagando por los mismos pesos; con los δ se
                    obtienen los gradientes ∂C/∂w = δ_destino · a_origen.
  4. ACTUALIZACIÓN: w ← w - η ∂C/∂w, y se ve cómo baja el costo.

Se puede avanzar paso a paso (Siguiente / Anterior), reproducir la iteración
completa o entrenar varias iteraciones seguidas para ver bajar el costo.

Pensado para Colab/Jupyter. Módulo autónomo (numpy / matplotlib / ipywidgets).
La lógica de cómputo (:func:`calcular`, :func:`actualizar`) está separada del
dibujo (:func:`figura_backprop`) y de los widgets (:func:`backprop_interactivo`).

Uso::

    from dlviz import backprop_interactivo
    backprop_interactivo()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Ejemplo y parámetros iniciales (fijos, para un demo reproducible)
# ---------------------------------------------------------------------------
X_EJ = np.array([0.6, 0.2])          # entrada
Y_EJ = np.array([0.9, 0.1])          # salida deseada (target)


def params_iniciales():
    """Pesos y sesgos iniciales de la red 2-2-2. W[j, i]: de la neurona i a la j."""
    return {
        "W1": np.array([[0.3, -0.4], [0.5, 0.7]]),
        "b1": np.array([0.1, -0.2]),
        "W2": np.array([[0.8, -0.5], [0.4, 0.9]]),
        "b2": np.array([0.1, -0.1]),
    }


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def calcular(p, x=X_EJ, y=Y_EJ):
    """Forward + costo + backward completos. Devuelve todos los valores intermedios."""
    z1 = p["W1"] @ x + p["b1"]
    a1 = _sig(z1)
    z2 = p["W2"] @ a1 + p["b2"]
    a2 = _sig(z2)
    e = a2 - y
    C = 0.5 * float((e ** 2).sum())
    d2 = e * a2 * (1 - a2)                       # δ de salida
    gW2 = np.outer(d2, a1)                       # ∂C/∂W2 = δ_o · a_h
    d1 = (p["W2"].T @ d2) * a1 * (1 - a1)        # δ oculto (retropropagado)
    gW1 = np.outer(d1, x)                        # ∂C/∂W1 = δ_h · x
    return {"x": x, "y": y, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "e": e,
            "C": C, "d2": d2, "d1": d1, "gW2": gW2, "gb2": d2.copy(),
            "gW1": gW1, "gb1": d1.copy()}


def actualizar(p, v, lr):
    """Un paso de descenso de gradiente: w ← w - η ∂C/∂w. Devuelve params nuevos."""
    return {"W1": p["W1"] - lr * v["gW1"], "b1": p["b1"] - lr * v["gb1"],
            "W2": p["W2"] - lr * v["gW2"], "b2": p["b2"] - lr * v["gb2"]}


# ---------------------------------------------------------------------------
# Guion de la animación: 13 pasos por iteración
# ---------------------------------------------------------------------------
PASOS = [
    {"fase": 0, "tipo": "entrada"},
    {"fase": 0, "tipo": "fwd_h", "k": 0},
    {"fase": 0, "tipo": "fwd_h", "k": 1},
    {"fase": 0, "tipo": "fwd_o", "k": 0},
    {"fase": 0, "tipo": "fwd_o", "k": 1},
    {"fase": 1, "tipo": "costo"},
    {"fase": 2, "tipo": "delta_o", "k": 0},
    {"fase": 2, "tipo": "delta_o", "k": 1},
    {"fase": 2, "tipo": "grad_2"},
    {"fase": 2, "tipo": "delta_h", "k": 0},
    {"fase": 2, "tipo": "delta_h", "k": 1},
    {"fase": 2, "tipo": "grad_1"},
    {"fase": 3, "tipo": "update"},
]
N_PASOS = len(PASOS)
FASES = ["1  FORWARD", "2  ERROR Y COSTO", "3  BACKWARD", "4  ACTUALIZACIÓN"]
C_FASE = ["#2471a3", "#e67e22", "#c0392b", "#1e8449"]
GRIS = "#b3bac2"

# Paso en que se revela cada cantidad
P_A1 = [1, 2]
P_A2 = [3, 4]
P_COSTO = 5
P_D2 = [6, 7]
P_G2 = 8
P_D1 = [9, 10]
P_G1 = 11
P_UPD = 12

# Geometría del diagrama
XL = [0.0, 3.3, 6.6]                 # x de cada capa
YN = [1.7, -1.7]                     # y de las dos neuronas de cada capa
X_T = 8.75                           # x de los targets
R = 0.46                             # radio de las neuronas


def _f(v, n=3):
    return f"{v:.{n}f}"


def _fp(v, n=3):
    """Número con paréntesis si es negativo (para sustituciones)."""
    s = f"{v:.{n}f}"
    return f"({s})" if v < 0 else s


# ---------------------------------------------------------------------------
# Dibujo de la red
# ---------------------------------------------------------------------------
def _neurona(ax, x, y, nombre, valor, color, activa=False, revelada=True,
             info=()):
    """Dibuja una neurona. `info` = lista de (texto, color, destacado) que se apila
    hacia afuera de la red (arriba si y>0, abajo si y<0), donde no hay aristas."""
    ec = color if activa else ("#33425b" if revelada else GRIS)
    fc = "#fdf2e9" if activa else ("white" if revelada else "#f3f4f6")
    ax.add_patch(mpatches.Circle((x, y), R, facecolor=fc, edgecolor=ec,
                 lw=3.2 if activa else 1.6, zorder=4))
    ax.text(x, y, valor if revelada else "?", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#222" if revelada else "#9aa0a6",
            zorder=5)
    s = 1 if y > 0 else -1
    yy = y + s * (R + 0.24)
    ax.text(x, yy, nombre, ha="center", va="center", fontsize=11, color="#222", zorder=5)
    for txt, colr, dest in info:
        yy += s * 0.3
        ax.text(x, yy, txt, ha="center", va="center", fontsize=8.6, color=colr,
                fontweight="bold" if dest else "normal", zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="#fdecea" if dest else "white",
                          ec="none", alpha=0.9))


def _punto_borde(p0, p1, r):
    d = np.subtract(p1, p0)
    d = d / np.linalg.norm(d)
    return (p0[0] + d[0] * r, p0[1] + d[1] * r), (p1[0] - d[0] * r, p1[1] - d[1] * r)


def _arista(ax, p0, p1, color, lw, flecha=None, alpha=1.0):
    a, b = _punto_borde(p0, p1, R + 0.02)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha, zorder=2,
            solid_capstyle="round")
    if flecha:  # "fwd": de p0 a p1 ; "bwd": de p1 a p0 (punta a media arista)
        f0, f1 = (0.42, 0.64) if flecha == "fwd" else (0.64, 0.42)
        pt = lambda t: (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        ax.annotate("", xy=pt(f1), xytext=pt(f0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=18), zorder=3)


def _etiqueta_peso(ax, p0, p1, texto, color, t=0.34, fc="white"):
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    ax.text(x, y, texto, ha="center", va="center", fontsize=7.9, color=color,
            zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc=fc, ec=color,
                                lw=0.8, alpha=0.95))


def _dibujar_red(ax, p, v, paso, p_nuevo):
    info = PASOS[paso]
    tipo, k = info["tipo"], info.get("k")
    col = C_FASE[info["fase"]]
    ax.set_xlim(-0.8, 9.6)
    ax.set_ylim(-3.95, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    pos_in = [(XL[0], YN[0]), (XL[0], YN[1])]
    pos_h = [(XL[1], YN[0]), (XL[1], YN[1])]
    pos_o = [(XL[2], YN[0]), (XL[2], YN[1])]

    # ---- Aristas capa 1 (x -> h) y capa 2 (h -> o) ----
    for capa, (orig, dest, W, gW) in enumerate(
            [(pos_in, pos_h, p["W1"], v["gW1"]), (pos_h, pos_o, p["W2"], v["gW2"])]):
        paso_grad = P_G1 if capa == 0 else P_G2
        for j in range(2):
            for i in range(2):
                c, lw, fl = GRIS, 1.4, None
                if capa == 0 and tipo == "fwd_h" and j == k:
                    c, lw, fl = col, 2.8, "fwd"
                elif capa == 1 and tipo == "fwd_o" and j == k:
                    c, lw, fl = col, 2.8, "fwd"
                elif capa == 1 and tipo == "delta_h" and i == k:
                    c, lw, fl = col, 2.8, "bwd"
                elif (capa == 0 and tipo == "grad_1") or (capa == 1 and tipo == "grad_2"):
                    c, lw = col, 2.4
                elif tipo == "update":
                    c, lw = col, 2.2
                _arista(ax, orig[i], dest[j], c, lw, fl)
                # etiqueta del peso (y su gradiente cuando ya se conoce)
                if tipo == "update":
                    nuevo = (p_nuevo["W1"] if capa == 0 else p_nuevo["W2"])[j, i]
                    txt, ec, fc = f"{W[j, i]:.3f}→{nuevo:.3f}", col, "#eafaf1"
                elif paso >= paso_grad:
                    txt = f"w = {W[j, i]:.2f}\n∂C/∂w = {gW[j, i]:+.4f}"
                    ec = "#c0392b" if paso == paso_grad else "#d98880"
                    fc = "#fdecea" if paso == paso_grad else "white"
                else:
                    activa_fwd = c != GRIS
                    txt, ec, fc = f"{W[j, i]:.2f}", (col if activa_fwd else "#7f8c8d"), "white"
                _etiqueta_peso(ax, orig[i], dest[j], txt, ec, fc=fc)

    # ---- Neuronas de entrada ----
    for i in range(2):
        act = (tipo == "entrada")
        _neurona(ax, *pos_in[i], f"$x_{i+1}$", _f(v["x"][i], 2), col, activa=act)

    # ---- Neuronas ocultas y de salida ----
    for capa, (pos, pref, z, a, d, bs, pa, pd, pg, bn) in enumerate([
            (pos_h, "h", v["z1"], v["a1"], v["d1"], p["b1"], P_A1, P_D1, P_G1, "b1"),
            (pos_o, "o", v["z2"], v["a2"], v["d2"], p["b2"], P_A2, P_D2, P_G2, "b2")]):
        for j in range(2):
            rev = paso >= pa[j]
            activa = ((tipo == ("fwd_h" if capa == 0 else "fwd_o") and j == k) or
                      (tipo == ("delta_h" if capa == 0 else "delta_o") and j == k))
            info = []
            if rev:
                info.append((f"z={z[j]:.3f}", "#2471a3", False))
            if tipo == "update":
                info.append((f"b: {bs[j]:.3f}→{p_nuevo[bn][j]:.3f}", C_FASE[3], False))
            elif paso >= pg:
                info.append((f"b={bs[j]:.2f}   ∂b={d[j]:+.4f}", "#c0392b", False))
            else:
                info.append((f"b={bs[j]:.2f}", "#566573", False))
            if paso >= pd[j]:
                info.append((f"δ={d[j]:+.4f}", "#c0392b", True))
            _neurona(ax, *pos[j], f"${pref}_{j+1}$", _f(a[j]), col, activa=activa,
                     revelada=rev, info=info)

    # ---- Targets y errores ----
    for kk in range(2):
        ax.add_patch(mpatches.FancyBboxPatch((X_T - 0.55, YN[kk] - 0.3), 1.1, 0.6,
                     boxstyle="round,pad=0.02,rounding_size=0.1", facecolor="#fef5e7",
                     edgecolor="#e67e22", lw=1.4, zorder=4))
        ax.text(X_T, YN[kk], f"$y_{kk+1}$={v['y'][kk]:.2f}", ha="center", va="center",
                fontsize=9.5, zorder=5)
        if paso >= P_COSTO:
            en_costo = tipo == "costo"
            en_delta = tipo == "delta_o" and kk == k
            cc = "#e67e22" if en_costo else ("#c0392b" if en_delta else "#f0b27a")
            ax.annotate("", xy=(XL[2] + R + 0.05, YN[kk]) if en_delta else (X_T - 0.58, YN[kk]),
                        xytext=(X_T - 0.58, YN[kk]) if en_delta else (XL[2] + R + 0.05, YN[kk]),
                        arrowprops=dict(arrowstyle="-|>", color=cc, lw=2.2 if (en_costo or en_delta) else 1.3,
                                        linestyle="--"), zorder=3)
            ax.text((XL[2] + R + X_T - 0.58) / 2, YN[kk] + 0.26, f"e={v['e'][kk]:+.3f}",
                    ha="center", fontsize=8.5, color=cc, fontweight="bold", zorder=5)

    # Rótulos de capa
    for xx, t in zip(XL + [X_T], ["entrada", "capa oculta", "capa de salida", "target"]):
        ax.text(xx, -3.8, t, ha="center", fontsize=9.5, color="#555", style="italic")

    if paso >= P_COSTO:
        ax.text(X_T, 0.0, f"C = {v['C']:.4f}", ha="center", va="center", fontsize=11,
                fontweight="bold", color="#b9570b" if tipo != "costo" else "white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="#e67e22" if tipo == "costo" else "#fef5e7", ec="#e67e22"))


# ---------------------------------------------------------------------------
# Panel de cálculo (texto + fórmulas del paso actual)
# ---------------------------------------------------------------------------
def _lineas_calculo(p, v, paso, lr, p_nuevo, C_nuevo):
    info = PASOS[paso]
    tipo, k = info["tipo"], info.get("k")
    x, a1, a2, z1, z2 = v["x"], v["a1"], v["a2"], v["z1"], v["z2"]
    L = []   # (texto, tamaño, color, negrita)

    if tipo == "entrada":
        L += [("Entrada: se presenta un ejemplo a la red", 12.5, C_FASE[0], True),
              ("Queremos que la red produzca y a partir de x.", 10.5, "#444", False),
              (f"$x = ({_f(x[0], 2)},\\ {_f(x[1], 2)})$", 13, "#222", False),
              (f"$y = ({_f(v['y'][0], 2)},\\ {_f(v['y'][1], 2)})$", 13, "#222", False),
              ("Los números sobre las conexiones son los pesos w;", 10.5, "#444", False),
              ("cada neurona tiene además un sesgo b.", 10.5, "#444", False),
              ("Activación: sigmoide  σ(z) = 1 / (1 + e^(−z))", 10.5, "#444", False)]
    elif tipo in ("fwd_h", "fwd_o"):
        oc = tipo == "fwd_h"
        W, b, ent = (p["W1"], p["b1"], x) if oc else (p["W2"], p["b2"], a1)
        z, a = (z1, a1) if oc else (z2, a2)
        n = f"h_{k+1}" if oc else f"o_{k+1}"
        cap = "(1)" if oc else "(2)"
        e_s = "x" if oc else "a_{h"
        cierre = "" if oc else "}"
        L += [(f"Forward: neurona ${n}$", 12.5, C_FASE[0], True),
              ("Suma ponderada de sus entradas + sesgo:", 10.5, "#444", False),
              (f"$z_{{{n}}} = w^{{{cap}}}_{{{k+1}1}}\\,{e_s}_1{cierre} + w^{{{cap}}}_{{{k+1}2}}\\,{e_s}_2{cierre} + b_{{{n}}}$",
               13, "#222", False),
              (f"$z_{{{n}}} = {_fp(W[k, 0], 2)}\\cdot{_f(ent[0])} + {_fp(W[k, 1], 2)}\\cdot{_f(ent[1])}"
               f" + {_fp(b[k], 2)} = {_f(z[k])}$", 12, "#222", False),
              ("Luego la activación (no linealidad):", 10.5, "#444", False),
              (f"$a_{{{n}}} = \\sigma(z_{{{n}}}) = \\sigma({_f(z[k])}) = {_f(a[k])}$", 13, "#222", False)]
        if not oc:
            L += [("Las entradas de esta capa son las activaciones", 10, "#666", False),
                  ("de la capa oculta calculadas antes.", 10, "#666", False)]
    elif tipo == "costo":
        e = v["e"]
        L += [("Error de cada salida y costo total", 12.5, C_FASE[1], True),
              ("Comparamos lo predicho con lo deseado:", 10.5, "#444", False),
              (f"$e_1 = a_{{o_1}} - y_1 = {_f(a2[0])} - {_f(v['y'][0], 2)} = {_f(e[0])}$", 12, "#222", False),
              (f"$e_2 = a_{{o_2}} - y_2 = {_f(a2[1])} - {_f(v['y'][1], 2)} = {_f(e[1])}$", 12, "#222", False),
              ("Costo (error cuadrático):", 10.5, "#444", False),
              (r"$C = \frac{1}{2}\sum_k (a_{o_k} - y_k)^2$", 13, "#222", False),
              (f"$C = \\frac{{1}}{{2}}\\,({_fp(e[0])}^2 + {_fp(e[1])}^2) = {v['C']:.4f}$", 12, "#222", False),
              ("El ½ simplifica la derivada: ∂C/∂a = a − y.", 10, "#666", False)]
    elif tipo == "delta_o":
        e, d = v["e"], v["d2"]
        L += [(f"Backward: δ de la salida $o_{k+1}$", 12.5, C_FASE[2], True),
              ("δ mide cuánto cambia C si cambia z de la neurona.", 10.5, "#444", False),
              ("Regla de la cadena:", 10.5, "#444", False),
              (f"$\\delta_{{o_{k+1}}} = \\frac{{\\partial C}}{{\\partial a_{{o_{k+1}}}}}"
               f"\\cdot\\frac{{\\partial a_{{o_{k+1}}}}}{{\\partial z_{{o_{k+1}}}}}"
               f" = (a_{{o_{k+1}}} - y_{k+1})\\,\\sigma'(z_{{o_{k+1}}})$", 13, "#222", False),
              (r"con  $\sigma'(z) = a\,(1-a)$", 11.5, "#444", False),
              (f"$\\delta_{{o_{k+1}}} = {_fp(e[k])}\\cdot{_f(a2[k])}\\cdot(1-{_f(a2[k])}) = {d[k]:+.4f}$",
               12, "#222", False),
              ("Signo +: la salida es demasiado alta → bajar z." if d[k] > 0 else
               "Signo −: la salida es demasiado baja → subir z.", 10.5, "#c0392b", False)]
    elif tipo in ("grad_2", "grad_1"):
        sal = tipo == "grad_2"
        d, g = (v["d2"], v["gW2"]) if sal else (v["d1"], v["gW1"])
        ent = a1 if sal else x
        nd, no = ("o", "a_{h") if sal else ("h", "x")
        cz = "}" if sal else ""
        cap = "(2)" if sal else "(1)"
        L += [(f"Backward: gradientes de la capa {'de salida' if sal else 'oculta'}", 12.5, C_FASE[2], True),
              ("Cada peso: δ de su neurona DESTINO × activación", 10.5, "#444", False),
              ("de su neurona ORIGEN.", 10.5, "#444", False),
              (f"$\\frac{{\\partial C}}{{\\partial w^{{{cap}}}_{{ji}}}} = \\delta_{{{nd}_j}}\\;{no}_i{cz}"
               f"\\qquad \\frac{{\\partial C}}{{\\partial b^{{{cap}}}_j}} = \\delta_{{{nd}_j}}$", 13, "#222", False)]
        for j in range(2):
            for i in range(2):
                L.append((f"$\\partial C/\\partial w_{{{j+1}{i+1}}} = {_fp(d[j], 4)}\\cdot{_f(ent[i])}"
                          f" = {g[j, i]:+.4f}$", 11, "#222", False))
        L.append(("Aparecen en rojo (∂C/∂w) sobre cada conexión.", 10, "#666", False))
    elif tipo == "delta_h":
        d2, W2, d1 = v["d2"], p["W2"], v["d1"]
        suma = W2[0, k] * d2[0] + W2[1, k] * d2[1]
        L += [(f"Backward: δ retropropagado a $h_{k+1}$", 12.5, C_FASE[2], True),
              ("h no tiene target propio: su culpa se obtiene de", 10.5, "#444", False),
              ("los δ de la salida, viajando hacia atrás por los", 10.5, "#444", False),
              ("MISMOS pesos que usó el forward.", 10.5, "#444", False),
              (f"$\\delta_{{h_{k+1}}} = \\left(w^{{(2)}}_{{1{k+1}}}\\,\\delta_{{o_1}} + "
               f"w^{{(2)}}_{{2{k+1}}}\\,\\delta_{{o_2}}\\right)\\sigma'(z_{{h_{k+1}}})$", 13, "#222", False),
              (f"$= ({_fp(W2[0, k], 2)}\\cdot{_fp(d2[0], 4)} + {_fp(W2[1, k], 2)}\\cdot{_fp(d2[1], 4)})"
               f"\\cdot{_f(a1[k])}\\,(1-{_f(a1[k])})$", 11.5, "#222", False),
              (f"$= {suma:+.4f}\\cdot{a1[k] * (1 - a1[k]):.4f} = {d1[k]:+.4f}$", 12, "#222", False)]
        if abs(d1[k]) < 0.01:
            L.append(("Los dos δ de salida casi se cancelan: δ pequeño.", 10, "#c0392b", False))
    else:  # update
        w_old, g, w_new = p["W2"][0, 0], v["gW2"][0, 0], p_nuevo["W2"][0, 0]
        mejora = C_nuevo < v["C"]
        L += [("Actualización: descenso de gradiente", 12.5, C_FASE[3], True),
              ("Cada peso se mueve CONTRA su gradiente:", 10.5, "#444", False),
              (f"$w \\leftarrow w - \\eta\\,\\frac{{\\partial C}}{{\\partial w}}\\qquad(\\eta = {lr:.2f})$",
               13, "#222", False),
              ("Ejemplo (peso h1 → o1):", 10.5, "#444", False),
              (f"${_f(w_old)} - {lr:.2f}\\cdot{_fp(g, 4)} = {_f(w_new)}$", 12, "#222", False),
              ("Nuevo forward con los pesos actualizados:", 10.5, "#444", False),
              (f"$C:\\ {v['C']:.4f}\\ \\rightarrow\\ {C_nuevo:.4f}$", 14,
               C_FASE[3] if mejora else "#c0392b", True),
              ("¡El costo bajó! Siguiente ▸ inicia otra iteración." if mejora else
               "El costo subió: η demasiado grande.", 10.5,
               C_FASE[3] if mejora else "#c0392b", False)]
    return L


def _dibujar_calculo(ax, lineas):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(mpatches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                 boxstyle="round,pad=0.0,rounding_size=0.03", transform=ax.transAxes,
                 facecolor="#fbfcfd", edgecolor="#d5dbe1", lw=1.2))
    y = 0.93
    for txt, fs, colr, neg in lineas:
        alto = 0.115 if ("\\frac" in txt or "\\sum" in txt or "\\left" in txt) else 0.085
        t = ax.text(0.04, y, txt, ha="left", va="top", fontsize=fs, color=colr,
                    fontweight="bold" if neg else "normal")
        t.set_in_layout(False)   # el layout no depende del texto: la figura no "salta"
        y -= alto * (fs / 11.5) + (0.02 if neg else 0.0)


# ---------------------------------------------------------------------------
# Figura completa
# ---------------------------------------------------------------------------
def figura_backprop(p=None, paso=0, lr=1.0, historial=None, iteracion=0):
    """Dibuja el paso `paso` (0..12) de una iteración de backprop con parámetros `p`.

    `historial` es la lista de costos por iteración (para la curva); se usa
    solo para dibujo, no se modifica.
    """
    p = params_iniciales() if p is None else p
    paso = int(np.clip(paso, 0, N_PASOS - 1))
    v = calcular(p)
    p_nuevo = actualizar(p, v, lr)
    C_nuevo = calcular(p_nuevo)["C"]
    historial = list(historial) if historial else [v["C"]]
    info = PASOS[paso]

    fig = plt.figure(figsize=(13, 7.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.09, 0.62, 0.29], width_ratios=[1.6, 1])
    ax_fase = fig.add_subplot(gs[0, :])
    ax_red = fig.add_subplot(gs[1:, 0])
    ax_calc = fig.add_subplot(gs[1, 1])
    ax_cost = fig.add_subplot(gs[2, 1])

    # ---- Barra de fases ----
    ax_fase.axis("off")
    ax_fase.set_xlim(0, 4)
    ax_fase.set_ylim(0, 1)
    for f, nom in enumerate(FASES):
        act = f == info["fase"]
        hecho = f < info["fase"]
        fc = C_FASE[f] if act else ("#e8ecef" if not hecho else "#d5dbe1")
        ax_fase.add_patch(mpatches.FancyBboxPatch((f + 0.04, 0.1), 0.92, 0.8,
                          boxstyle="round,pad=0.0,rounding_size=0.12",
                          facecolor=fc, edgecolor=C_FASE[f], lw=2 if act else 1))
        ax_fase.text(f + 0.5, 0.5, nom + ("  ✓" if hecho else ""), ha="center", va="center",
                     fontsize=10.5, fontweight="bold",
                     color="white" if act else ("#566573" if hecho else "#8d99a6"))

    _dibujar_red(ax_red, p, v, paso, p_nuevo)
    _dibujar_calculo(ax_calc, _lineas_calculo(p, v, paso, lr, p_nuevo, C_nuevo))

    # ---- Curva del costo por iteración ----
    its = np.arange(len(historial))
    ax_cost.plot(its, historial, "o-", color="#e67e22", ms=4, lw=1.6)
    ax_cost.plot([iteracion], [historial[iteracion]], "o", ms=9, mfc="none",
                 mec="#e67e22", mew=2)
    if info["tipo"] == "update":
        ax_cost.plot([iteracion, iteracion + 1], [v["C"], C_nuevo], "--", color=C_FASE[3], lw=1.4)
        ax_cost.plot([iteracion + 1], [C_nuevo], "o", ms=7, color=C_FASE[3])
    ax_cost.set_xlim(-0.5, max(5, len(historial)) + 0.5)
    ax_cost.set_ylim(0, max(historial + [C_nuevo]) * 1.15)
    ax_cost.set_xlabel("iteración", fontsize=9)
    ax_cost.set_ylabel("costo C", fontsize=9)
    ax_cost.tick_params(labelsize=8)
    ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_cost.grid(alpha=0.3)
    ax_cost.set_title("Costo a lo largo de las iteraciones", fontsize=10, fontweight="bold")

    fig.suptitle(f"Backpropagation paso a paso  ·  iteración {iteracion + 1}  ·  "
                 f"paso {paso + 1}/{N_PASOS}", fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Widget interactivo
# ---------------------------------------------------------------------------
def backprop_interactivo(lr=1.0, pausa=1.0):
    """Despliega el demo interactivo de backpropagation en un MLP 2-2-2.

    Controles:
      * ◀ Anterior / Siguiente ▶ : avanzar o retroceder un paso.
      * ▶ Reproducir iteración    : anima los pasos restantes de la iteración.
      * ⏩ Entrenar 20 iter.       : repite el proceso 20 veces y muestra cómo
                                     baja el costo.
      * ↺ Reiniciar               : vuelve a los pesos iniciales.
      * η                         : tasa de aprendizaje de la actualización.
    """
    import time
    import ipywidgets as widgets
    from IPython.display import display

    estado = {"hist_p": [params_iniciales()], "it": 0, "paso": 0}
    out = widgets.Output(layout=widgets.Layout(height="740px", overflow="hidden"))

    def costos():
        return [calcular(pp)["C"] for pp in estado["hist_p"]]

    def redibujar():
        with out:
            out.clear_output(wait=True)
            fig = figura_backprop(estado["hist_p"][estado["it"]], estado["paso"],
                                  s_lr.value, costos(), estado["it"])
            plt.show()
            plt.close(fig)

    def avanzar():
        """Un paso adelante; tras la actualización, empieza la siguiente iteración."""
        if estado["paso"] < N_PASOS - 1:
            estado["paso"] += 1
        else:
            p = estado["hist_p"][estado["it"]]
            nuevo = actualizar(p, calcular(p), s_lr.value)
            estado["hist_p"] = estado["hist_p"][:estado["it"] + 1] + [nuevo]
            estado["it"] += 1
            estado["paso"] = 0

    s_lr = widgets.FloatSlider(value=lr, min=0.1, max=5.0, step=0.1, description="η",
                               style={"description_width": "20px"},
                               layout=widgets.Layout(width="260px"), readout_format=".1f",
                               continuous_update=False)
    b_prev = widgets.Button(description="◀ Anterior", layout=widgets.Layout(width="110px"))
    b_next = widgets.Button(description="Siguiente ▶", button_style="info",
                            layout=widgets.Layout(width="120px"))
    b_play = widgets.Button(description="▶ Reproducir iteración", button_style="success",
                            layout=widgets.Layout(width="190px"))
    b_train = widgets.Button(description="⏩ Entrenar 20 iter.", button_style="warning",
                             layout=widgets.Layout(width="170px"))
    b_reset = widgets.Button(description="↺ Reiniciar", layout=widgets.Layout(width="110px"))
    todos = (b_prev, b_next, b_play, b_train, b_reset, s_lr)

    def on_prev(_):
        if estado["paso"] > 0:
            estado["paso"] -= 1
        elif estado["it"] > 0:
            estado["it"] -= 1
            estado["paso"] = N_PASOS - 1
        redibujar()

    def on_next(_):
        avanzar()
        redibujar()

    def bloquear(flag):
        for w in todos:
            w.disabled = flag

    def on_play(_):
        bloquear(True)
        try:
            if estado["paso"] == N_PASOS - 1:   # si ya terminó, empieza la siguiente
                avanzar()
            redibujar()
            time.sleep(pausa)
            while estado["paso"] < N_PASOS - 1:
                estado["paso"] += 1
                redibujar()
                time.sleep(pausa)
        finally:
            bloquear(False)

    def on_train(_):
        bloquear(True)
        try:
            p = estado["hist_p"][estado["it"]]
            hist = estado["hist_p"][:estado["it"] + 1]
            for _k in range(20):
                p = actualizar(p, calcular(p), s_lr.value)
                hist.append(p)
            estado["hist_p"] = hist
            estado["it"] = len(hist) - 1
            estado["paso"] = P_COSTO
            redibujar()
        finally:
            bloquear(False)

    def on_reset(_):
        estado.update({"hist_p": [params_iniciales()], "it": 0, "paso": 0})
        redibujar()

    b_prev.on_click(on_prev)
    b_next.on_click(on_next)
    b_play.on_click(on_play)
    b_train.on_click(on_train)
    b_reset.on_click(on_reset)
    s_lr.observe(lambda _: redibujar(), names="value")

    titulo = widgets.HTML(
        "<h3 style='margin-bottom:4px'>Backpropagation paso a paso</h3>"
        "<span style='color:#555'>Una red 2-2-2 (sigmoide) aprende con un ejemplo. "
        "<b style='color:#2471a3'>Forward</b>: cada neurona calcula z y su activación; "
        "<b style='color:#e67e22'>error y costo</b>; "
        "<b style='color:#c0392b'>backward</b>: los δ viajan hacia atrás por la regla de "
        "la cadena y dan los gradientes; "
        "<b style='color:#1e8449'>actualización</b>: w ← w − η ∂C/∂w. "
        "Avanza con <b>Siguiente</b> o reproduce la iteración completa.</span>")
    controles = widgets.HBox([b_prev, b_next, b_play, b_train, b_reset, s_lr])

    redibujar()
    display(widgets.VBox([titulo, controles, out]))
