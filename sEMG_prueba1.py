# %% [markdown]
# # sEMG + Torque — marcación manual de onset/offset de burst
#
# Versión para VS Code + Python Interactive/Jupyter (`%matplotlib widget`).
#
# ## Idea de la interfaz
# Una sola figura, con tres paneles que comparten el eje X:
#
#   1. Posición (tira delgada, solo orientación)
#   2. Torque  (SIEMPRE visible)  -> clic = onset de torque
#   3. sEMG del canal ACTIVO      -> clic = onset, segundo clic = offset
#
# Los 4 músculos no se apilan: se ve UNO a la vez y se cambia de canal con
# botones/teclas. Para comparar tiempos entre músculos, en el panel de sEMG
# quedan dibujados (líneas punteadas de color) los onsets ya marcados en los
# otros canales, y la línea roja del onset de torque cruza los tres paneles.
#
# La señal del equipo (velocidad) se usa SOLO para delimitar cada movimiento
# y paginar la ventana. Ningún onset/offset de sEMG ni de torque se calcula
# automáticamente: todo es manual.
#
# ## Controles
#   Clic en panel de TORQUE ........ onset de torque (otro clic lo reemplaza)
#   Clic en panel de sEMG .......... 1.er clic onset, 2.º clic offset,
#                                    3.er clic reinicia (nuevo onset)
#   Botones 1-4 / teclas 1-4 ....... cambiar de canal
#   "Sin burst" (b) ................ el músculo no se activó en esta contracción
#   "Rehacer canal" (u) ............ borra onset/offset del canal activo
#   "Escala Y" (y) ................. amplifica la señal para ver el inicio
#   "Envolvente" (e) ............... superpone envolvente RMS (solo ayuda visual)
#   "Contr. ◀ ▶" (a / d) ........... cambiar de contracción
#   "Pendiente ▶" (n) .............. salta a lo próximo sin marcar
#
# Si los clics no marcan: desactiva zoom/pan en la barra de herramientas.
# Las marcas se autoguardan en un JSON tras cada clic (puedes retomar).


# %% Configuración

import os
import json

FILE_PATH = r"C:\RUTA\AL\ARCHIVO\2026-08-19-11-09_HUMAC_NORM___sEMG_60_-_5reps.csv"

OUTPUT_DIR = r"C:\Users\crist\OneDrive\Documentos\Laboratorio Regulación Masa y Fuerza UFRO\Isocinética - datos en CSV\Proyecto Congreso\resultados"

# --- Columnas del CSV (exportación Noraxon myoRESEARCH) ---
TIME_COL = "time"
TORQUE_COL = "Torque (N*m)"
POSITION_COL = "Position (deg)"
VELOCITY_COL = "Velocity (deg/s)"

# Canales sEMG: etiqueta corta -> nombre de columna.
# El orden define el orden de los botones 1-4. Puedes quitar canales.
EMG_CHANNELS = {
    "RF_DER": "RECTUS FEM. RT (uV)",
    "BF_DER": "BICEPS FEM. RT (uV)",
    "RF_IZQ": "RECTUS FEM. LT (uV)",
    "BF_IZQ": "BICEPS FEM. LT (uV)",
}

# --- Segmentación de movimientos (solo para paginar) ---
VEL_THRESHOLD_DEG_S = 10      # |velocidad| > umbral => hay movimiento
MIN_MOVEMENT_S = 0.3          # descarta tramos más cortos
MERGE_GAP_S = 0.05            # une tramos del mismo sentido separados por < 50 ms

# Velocidad positiva = Extensión (en este archivo velocidad+ coincide con
# posición decreciente y torque pico ~200 N·m). Verifícalo con tu protocolo.
VEL_POSITIVE_IS_EXTENSION = True

# Movimientos pasivos (la máquina mueve la pierna sin esfuerzo): se detectan
# por variación de torque dentro del tramo y se excluyen automáticamente.
AUTO_EXCLUDE_PASSIVE = True
MIN_TORQUE_PTP_NM = 40

# Exclusión manual: IDs ("mov #") de la tabla impresa en la celda de detección.
EXCLUDE_MOVEMENTS = []

# --- Ventana visual alrededor de cada movimiento ---
# El músculo puede activarse ANTES del movimiento: margen previo generoso.
PRE_MARGIN_S = 0.5
POST_MARGIN_S = 0.3

# --- Visualización sEMG ---
ENVELOPE_WINDOW_MS = 50       # RMS móvil (solo ayuda visual)
Y_SCALE_STEPS = [1.0, 0.5, 0.25, 0.10, 0.05]   # fracción del máximo de la ventana

# --- Autoguardado / salida ---
RESUME = True                 # retomar marcas previas si existe el JSON
MAKE_REVIEW_PLOTS = False     # PNG de revisión con los 4 canales apilados

_stem = os.path.splitext(os.path.basename(FILE_PATH))[0]
AUTOSAVE_PATH = os.path.join(OUTPUT_DIR, f"marcas_autosave_{_stem}.json")
BURST_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "EMG_burst_por_canal.csv")
SUMMARY_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "EMG_resumen_por_contraccion.csv")
REVIEW_DIR = os.path.join(OUTPUT_DIR, "revision_emg")


# %% Backend interactivo

%matplotlib widget

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button

try:
    display
except NameError:  # fuera de Jupyter/Interactive
    def display(obj):
        print(obj)

print("Matplotlib:", mpl.__version__)
print("Backend:", mpl.get_backend())


# %% Cargar CSV (con cabecera de metadatos Noraxon) y validar

def find_header_line(path, encoding):
    """Línea (0-based) donde empieza la tabla de datos (primer campo = 'time')."""
    with open(path, "r", encoding=encoding) as fh:
        for i, line in enumerate(fh):
            first = line.lstrip("\ufeff").split(",")[0].strip().strip('"').lower()
            if first == "time":
                return i
    raise ValueError("No se encontró la fila de cabecera que empieza con 'time'.")


df = None
for enc in ("utf-8-sig", "latin1"):
    try:
        header_line = find_header_line(FILE_PATH, enc)
        df = pd.read_csv(FILE_PATH, skiprows=header_line, encoding=enc)
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise ValueError("No se pudo leer el CSV con utf-8 ni latin1.")

required_cols = [TIME_COL, TORQUE_COL, POSITION_COL, VELOCITY_COL] + list(EMG_CHANNELS.values())
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        "Faltan columnas en el CSV:\n" + "\n".join(f" - {c}" for c in missing)
        + "\n\nColumnas disponibles:\n" + "\n".join(f" - {c}" for c in df.columns)
    )

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

n_before = len(df)
df = df.dropna(subset=required_cols).reset_index(drop=True)
if len(df) < n_before:
    print(f"⚠️ Se descartaron {n_before - len(df)} filas con valores no numéricos.")

time = df[TIME_COL].to_numpy(float)
torque = df[TORQUE_COL].to_numpy(float)
position = df[POSITION_COL].to_numpy(float)
velocity = df[VELOCITY_COL].to_numpy(float)

if not np.all(np.diff(time) > 0):
    raise ValueError("La columna de tiempo no es estrictamente creciente.")

FS = 1.0 / np.median(np.diff(time))

# sEMG: se quita el offset DC (mediana de todo el registro)
emg = {}
for key, col in EMG_CHANNELS.items():
    x = df[col].to_numpy(float)
    emg[key] = x - np.median(x)


def moving_rms(x, window_ms):
    w = max(3, int(round(window_ms / 1000 * FS)))
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same"))


envelope = {k: moving_rms(x, ENVELOPE_WINDOW_MS) for k, x in emg.items()}

print(f"Archivo: {os.path.basename(FILE_PATH)}")
print(f"Muestras: {len(df):,}  |  Duración: {time[-1] - time[0]:.2f} s  |  Fs ≈ {FS:.0f} Hz")
print("Canales sEMG:", ", ".join(EMG_CHANNELS))


# %% Detectar movimientos con la velocidad (solo para paginar)

def detect_movements():
    s = np.zeros(len(velocity), dtype=int)
    s[velocity > VEL_THRESHOLD_DEG_S] = 1
    s[velocity < -VEL_THRESHOLD_DEG_S] = -1

    change = np.flatnonzero(np.diff(s)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, len(s) - 1]
    runs = [(a, b, s[a]) for a, b in zip(starts, ends) if s[a] != 0]

    merged = []
    for a, b, sg in runs:
        if merged and merged[-1][2] == sg and (time[a] - time[merged[-1][1]]) < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], b, sg)
        else:
            merged.append((a, b, sg))

    return [(a, b, sg) for a, b, sg in merged if time[b] - time[a] >= MIN_MOVEMENT_S]


movements = []
for mov_id, (a, b, sg) in enumerate(detect_movements(), start=1):
    is_ext = (sg > 0) == VEL_POSITIVE_IS_EXTENSION
    ptp = float(np.ptp(torque[a:b + 1]))
    passive = ptp < MIN_TORQUE_PTP_NM
    manual_excl = mov_id in EXCLUDE_MOVEMENTS

    if manual_excl:
        status = "excluido (manual)"
    elif passive and AUTO_EXCLUDE_PASSIVE:
        status = "excluido (pasivo)"
    else:
        status = "INCLUIDO"

    movements.append({
        "mov_id": mov_id,
        "movimiento": "Extensión" if is_ext else "Flexión",
        "start_idx": int(a),
        "end_idx": int(b),
        "start_time": float(time[a]),
        "end_time": float(time[b]),
        "pos_ini": float(position[a]),
        "pos_fin": float(position[b]),
        "torque_ptp": ptp,
        "incluir": status == "INCLUIDO",
        "estado": status,
    })

print(f"Movimientos detectados: {len(movements)}\n")
print(" mov | movimiento |   inicio →   fin (s) | pos ini → fin (°) | ΔTorque (N·m) | estado")
print("-" * 92)
for m in movements:
    print(
        f" {m['mov_id']:>3} | {m['movimiento']:<10} | "
        f"{m['start_time']:>8.3f} → {m['end_time']:>7.3f}   | "
        f"{m['pos_ini']:>6.1f} → {m['pos_fin']:>6.1f}   | "
        f"{m['torque_ptp']:>11.1f}   | {m['estado']}"
    )

contractions = []
for m in [m for m in movements if m["incluir"]]:
    n = len(contractions) + 1
    contractions.append({**m, "n": n, "repeticion": (n + 1) // 2})

print(f"\nContracciones a marcar: {len(contractions)}")
if len(contractions) != 10:
    print(
        "⚠️ No son 10. Revisa la tabla (p. ej. un movimiento parcial al inicio/fin) "
        "y usa EXCLUDE_MOVEMENTS en la celda de configuración si corresponde."
    )
if not contractions:
    raise RuntimeError("No hay contracciones para marcar: revisa los parámetros de segmentación.")


# %% Estado de marcas + autoguardado

N = len(contractions)
ch_keys = list(EMG_CHANNELS.keys())
n_ch = len(ch_keys)
cont_by_n = {c["n"]: c for c in contractions}
CH_COLORS = {k: plt.cm.tab10(i) for i, k in enumerate(ch_keys)}

marks = {}


def entry(cn):
    if cn not in marks:
        marks[cn] = {
            "torque_onset": None,
            "ch": {k: {"onset": None, "offset": None, "no_burst": False} for k in ch_keys},
        }
    return marks[cn]


def ch_done(cn, k):
    e = entry(cn)["ch"][k]
    return e["no_burst"] or (e["onset"] is not None and e["offset"] is not None)


def save_marks():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        payload = {
            "archivo": os.path.basename(FILE_PATH),
            "marks": {
                str(cn): {**m, "start_time": cont_by_n[cn]["start_time"]}
                for cn, m in marks.items() if cn in cont_by_n
            },
        }
        with open(AUTOSAVE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
    except Exception as exc:  # el autoguardado nunca debe romper la interfaz
        print(f"⚠️ No se pudo autoguardar: {exc}")


def load_marks():
    if not (RESUME and os.path.exists(AUTOSAVE_PATH)):
        return
    try:
        with open(AUTOSAVE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"⚠️ No se pudo leer el autoguardado: {exc}")
        return
    if payload.get("archivo") != os.path.basename(FILE_PATH):
        print("⚠️ El autoguardado es de otro archivo; se ignora.")
        return

    loaded = 0
    for cn_str, m in payload.get("marks", {}).items():
        cn = int(cn_str)
        if cn not in cont_by_n or abs(m.get("start_time", -1) - cont_by_n[cn]["start_time"]) > 0.05:
            print(f"⚠️ Contracción {cn} del autoguardado no coincide con la segmentación actual; se ignora.")
            continue
        e = entry(cn)
        e["torque_onset"] = m.get("torque_onset")
        for k in ch_keys:
            if k in m.get("ch", {}):
                e["ch"][k].update(m["ch"][k])
        loaded += 1
    print(f"↻ Se retomaron marcas de {loaded} contracciones desde:\n  {AUTOSAVE_PATH}")


load_marks()


# %% SELECCIÓN MANUAL — una figura, un canal sEMG a la vez

plt.close("all")

cur = {"i": 0, "ch": 0, "env": False, "y": 0, "msg": ""}

fig = plt.figure(figsize=(12, 8.6))

ax_pos = fig.add_axes([0.07, 0.815, 0.90, 0.10])
ax_tq = fig.add_axes([0.07, 0.590, 0.90, 0.210], sharex=ax_pos)
ax_emg = fig.add_axes([0.07, 0.345, 0.90, 0.230], sharex=ax_pos)
axes_all = (ax_pos, ax_tq, ax_emg)

title_text = fig.text(0.07, 0.978, "", ha="left", va="top", fontsize=11, fontweight="bold")
status_text = fig.text(0.07, 0.950, "", ha="left", va="top", fontsize=10)
info_text = fig.text(
    0.07, 0.272, "", ha="left", va="top", fontsize=8.8, family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", alpha=0.08),
)

# Fila A: canales + escala Y
btn_ch = []
for j, k in enumerate(ch_keys):
    b = Button(fig.add_axes([0.07 + j * 0.185, 0.078, 0.16, 0.043]), f"{j + 1} · {k}")
    b.label.set_fontsize(9.5)
    btn_ch.append(b)
btn_y = Button(fig.add_axes([0.07 + n_ch * 0.185, 0.078, 0.16, 0.043]), "")
btn_y.label.set_fontsize(9)

# Fila B: acciones
_row_b = [
    ("prev", "◀ Contr. (a)"), ("next", "Contr. ▶ (d)"), ("redo", "Rehacer canal (u)"),
    ("nob", "Sin burst (b)"), ("env", "Envolvente (e)"), ("pend", "Pendiente ▶ (n)"),
]
btn = {}
for j, (name, label) in enumerate(_row_b):
    b = Button(fig.add_axes([0.07 + j * 0.152, 0.018, 0.140, 0.043]), label)
    b.label.set_fontsize(9)
    btn[name] = b


# ---------- utilidades ----------

def nearest_idx(t):
    i = int(np.searchsorted(time, t))
    i = min(max(i, 1), len(time) - 1)
    return i - 1 if abs(time[i - 1] - t) <= abs(time[i] - t) else i


def fmt(v, f):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def is_pending(i, j):
    cn = contractions[i]["n"]
    return (not ch_done(cn, ch_keys[j])) or (j == 0 and entry(cn)["torque_onset"] is None)


def progress():
    n_t = sum(entry(c["n"])["torque_onset"] is not None for c in contractions)
    n_c = sum(ch_done(c["n"], k) for c in contractions for k in ch_keys)
    return n_t, n_c


def style_buttons():
    c = contractions[cur["i"]]
    for j, k in enumerate(ch_keys):
        done = ch_done(c["n"], k)
        btn_ch[j].label.set_text(f"{j + 1} · {k}" + (" ✓" if done else ""))
        active = j == cur["ch"]
        color = "#9ecbff" if active else ("#d7f0d7" if done else "0.85")
        btn_ch[j].color = color
        btn_ch[j].hovercolor = "#b8dcff" if active else "0.95"
        btn_ch[j].ax.set_facecolor(color)
    btn_y.label.set_text(f"Escala Y: {Y_SCALE_STEPS[cur['y']]:.0%} (y)")
    btn["env"].label.set_text("Envolvente: " + ("ON" if cur["env"] else "OFF") + " (e)")
    env_color = "#d7f0d7" if cur["env"] else "0.85"
    btn["env"].color = env_color
    btn["env"].ax.set_facecolor(env_color)


def update_info(c, e, k):
    cn = c["n"]
    t_tq = e["torque_onset"]
    lines = []
    if t_tq is None:
        lines.append("Torque onset: — sin marcar (clic en el panel de TORQUE) —")
    else:
        lines.append(
            f"Torque onset: {t_tq:.4f} s   ({(t_tq - c['start_time']) * 1000:+.0f} ms vs inicio de movimiento)"
        )
    lines.append(f"{'Canal':<8}{'onset (s)':>11}{'offset (s)':>12}{'dur (ms)':>10}{'EMG−torque (ms)':>17}")
    for j, k2 in enumerate(ch_keys):
        ch = e["ch"][k2]
        pref = "▶" if j == cur["ch"] else " "
        if ch["no_burst"]:
            row = f"{pref}{k2:<7}{'sin burst':>11}"
        else:
            on, off = ch["onset"], ch["offset"]
            dur = (off - on) * 1000 if (on is not None and off is not None) else None
            lat = (on - t_tq) * 1000 if (on is not None and t_tq is not None) else None
            row = (
                f"{pref}{k2:<7}{fmt(on, '{:.4f}'):>11}{fmt(off, '{:.4f}'):>12}"
                f"{fmt(dur, '{:.0f}'):>10}{fmt(lat, '{:+.1f}'):>17}"
            )
        lines.append(row + ("  ✓" if ch_done(cn, k2) else ""))
    lines.append("EMG−torque < 0  →  el músculo se activa ANTES que el torque.")
    info_text.set_text("\n".join(lines))


def update_plot():
    c = contractions[cur["i"]]
    cn = c["n"]
    k = ch_keys[cur["ch"]]
    e = entry(cn)
    ch = e["ch"][k]

    t0 = max(time[0], c["start_time"] - PRE_MARGIN_S)
    t1 = min(time[-1], c["end_time"] + POST_MARGIN_S)
    i0, i1 = int(np.searchsorted(time, t0)), int(np.searchsorted(time, t1))
    sl = slice(i0, i1 + 1)
    tt = time[sl]

    for ax in axes_all:
        ax.clear()

    # --- señales ---
    ax_pos.plot(tt, position[sl], color="0.35", lw=1.0)
    ax_tq.plot(tt, torque[sl], color="tab:blue", lw=1.2)

    x = emg[k][sl]
    ax_emg.plot(tt, x, color=CH_COLORS[k], lw=0.6)
    if cur["env"]:
        ax_emg.plot(tt, envelope[k][sl], color="k", lw=1.4, alpha=0.75, label=f"Envolvente RMS {ENVELOPE_WINDOW_MS} ms")
    lim = max(float(np.max(np.abs(x))) * 1.05, 1.0) * Y_SCALE_STEPS[cur["y"]]
    ax_emg.set_ylim(-lim, lim)

    # --- límites del movimiento según velocidad del equipo (solo referencia) ---
    for ax in axes_all:
        ax.axvline(c["start_time"], color="0.45", ls=":", lw=1.3)
        ax.axvline(c["end_time"], color="0.45", ls=":", lw=1.3)
    ax_tq.plot([], [], color="0.45", ls=":", lw=1.3, label=f"Inicio/fin movimiento (|vel|>{VEL_THRESHOLD_DEG_S}°/s)")

    # --- onset de torque (cruza los 3 paneles) ---
    if e["torque_onset"] is not None:
        t_tq = e["torque_onset"]
        for ax in axes_all:
            ax.axvline(t_tq, color="red", ls="--", lw=1.8)
        ax_tq.plot(t_tq, torque[nearest_idx(t_tq)], "o", color="red", ms=7)
        ax_tq.plot([], [], color="red", ls="--", lw=1.8, label="Onset torque (manual)")

    # --- canal activo ---
    if ch["onset"] is not None:
        ax_emg.axvline(ch["onset"], color="tab:green", lw=2.0, label=f"Onset {k}")
    if ch["offset"] is not None:
        ax_emg.axvline(ch["offset"], color="darkorange", lw=2.0, label=f"Offset {k}")
    if ch["onset"] is not None and ch["offset"] is not None:
        ax_emg.axvspan(ch["onset"], ch["offset"], color="tab:green", alpha=0.12)

    # --- onsets de los otros canales (referencia para comparar músculos) ---
    for k2 in ch_keys:
        if k2 == k:
            continue
        on2 = e["ch"][k2]["onset"]
        if on2 is not None and t0 <= on2 <= t1:
            ax_emg.axvline(on2, color=CH_COLORS[k2], ls=":", lw=1.3, alpha=0.9)
            ax_emg.text(on2, 0.97, k2, transform=ax_emg.get_xaxis_transform(), rotation=90,
                        ha="right", va="top", fontsize=7, color=CH_COLORS[k2])

    # --- formato ---
    ax_pos.set_ylabel("Pos (°)")
    ax_tq.set_ylabel("Torque (N·m)")
    ax_emg.set_ylabel(f"{k} (µV)")
    ax_emg.set_xlabel("Tiempo (s)")
    for ax in axes_all:
        ax.grid(alpha=0.3)
    ax_pos.tick_params(labelbottom=False)
    ax_tq.tick_params(labelbottom=False)
    ax_tq.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
    if cur["env"]:
        ax_emg.legend(loc="upper right", fontsize=7.5, framealpha=0.7)
    ax_emg.set_xlim(t0, t1)

    # --- textos ---
    title_text.set_text(
        f"Contracción {cur['i'] + 1}/{N} · Rep {c['repeticion']} · {c['movimiento']} · "
        f"mov #{c['mov_id']} · {c['start_time']:.3f}–{c['end_time']:.3f} s · Canal activo: {k}"
    )

    if ch["no_burst"]:
        hint = f"{k} marcado SIN BURST ('Rehacer canal' para cambiarlo)."
    elif ch["onset"] is None:
        hint = f"Próximo clic en sEMG ({k}): ONSET."
    elif ch["offset"] is None:
        hint = f"Próximo clic en sEMG ({k}): OFFSET."
    else:
        hint = f"{k} completo. Un clic nuevo reinicia el canal (nuevo onset)."
    if e["torque_onset"] is None:
        hint += "  Falta onset de torque."
    n_t, n_c = progress()
    status_text.set_text(
        (cur["msg"] + "   " if cur["msg"] else "")
        + hint + f"   | Progreso: torque {n_t}/{N} · canales {n_c}/{N * n_ch}"
    )
    cur["msg"] = ""

    update_info(c, e, k)
    style_buttons()
    fig.canvas.draw_idle()


# ---------- eventos ----------

def on_click(event):
    if getattr(event, "button", None) != 1 or getattr(event, "dblclick", False):
        return
    toolbar = getattr(fig.canvas, "toolbar", None)
    if toolbar is not None and str(getattr(toolbar, "mode", "")) != "":
        return  # zoom/pan activo: no marcar
    if event.inaxes not in axes_all or event.xdata is None:
        return

    c = contractions[cur["i"]]
    e = entry(c["n"])
    k = ch_keys[cur["ch"]]
    t_snap = float(time[nearest_idx(float(event.xdata))])

    if event.inaxes is ax_tq:
        e["torque_onset"] = t_snap
    elif event.inaxes is ax_emg:
        ch = e["ch"][k]
        ch["no_burst"] = False
        if ch["onset"] is None or ch["offset"] is not None or t_snap <= ch["onset"]:
            ch["onset"], ch["offset"] = t_snap, None
        else:
            ch["offset"] = t_snap
    else:
        cur["msg"] = "Marca en el panel de TORQUE o de sEMG."
        update_plot()
        return

    save_marks()
    update_plot()


def go_contraction(delta):
    new = cur["i"] + delta
    if 0 <= new < N:
        cur["i"] = new
        update_plot()


def set_channel(j):
    if 0 <= j < n_ch:
        cur["ch"] = j
        update_plot()


def redo_channel(_=None):
    ch = entry(contractions[cur["i"]]["n"])["ch"][ch_keys[cur["ch"]]]
    ch.update(onset=None, offset=None, no_burst=False)
    save_marks()
    update_plot()


def toggle_no_burst(_=None):
    ch = entry(contractions[cur["i"]]["n"])["ch"][ch_keys[cur["ch"]]]
    ch["no_burst"] = not ch["no_burst"]
    if ch["no_burst"]:
        ch["onset"], ch["offset"] = None, None
    save_marks()
    update_plot()


def toggle_env(_=None):
    cur["env"] = not cur["env"]
    update_plot()


def cycle_y(_=None):
    cur["y"] = (cur["y"] + 1) % len(Y_SCALE_STEPS)
    update_plot()


def next_pending(_=None):
    total = N * n_ch
    start = cur["i"] * n_ch + cur["ch"]
    for step in range(1, total + 1):
        i, j = divmod((start + step) % total, n_ch)
        if is_pending(i, j):
            cur["i"], cur["ch"] = i, j
            update_plot()
            return
    cur["msg"] = "✓ Todo marcado. Ejecuta la siguiente celda."
    update_plot()


def on_key(event):
    k = event.key
    if k == "a":
        go_contraction(-1)
    elif k == "d":
        go_contraction(+1)
    elif k == "n":
        next_pending()
    elif k == "u":
        redo_channel()
    elif k == "b":
        toggle_no_burst()
    elif k == "e":
        toggle_env()
    elif k == "y":
        cycle_y()
    elif k in [str(j + 1) for j in range(n_ch)]:
        set_channel(int(k) - 1)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

for j, b in enumerate(btn_ch):
    b.on_clicked(lambda _ev, j=j: set_channel(j))
btn_y.on_clicked(cycle_y)
btn["prev"].on_clicked(lambda _ev: go_contraction(-1))
btn["next"].on_clicked(lambda _ev: go_contraction(+1))
btn["redo"].on_clicked(redo_channel)
btn["nob"].on_clicked(toggle_no_burst)
btn["env"].on_clicked(toggle_env)
btn["pend"].on_clicked(next_pending)

# Si se retomó un trabajo previo, empezar en lo primero pendiente.
if not is_pending(0, 0):
    next_pending()
else:
    update_plot()

plt.show()


# %% Comprobar que todo esté marcado

problems = []
for c in contractions:
    cn = c["n"]
    if entry(cn)["torque_onset"] is None:
        problems.append(f"Contracción {cn}: falta onset de torque")
    for k in ch_keys:
        if not ch_done(cn, k):
            problems.append(f"Contracción {cn}, canal {k}: falta onset/offset (o marcar 'Sin burst')")

if problems:
    raise RuntimeError(
        f"Faltan {len(problems)} marcas. Vuelve a la celda de selección ('Pendiente ▶'):\n  "
        + "\n  ".join(problems)
    )

print(f"✓ {N} contracciones × {n_ch} canales marcados.")


# %% Calcular latencias, duración y RMS del burst

archivo = os.path.basename(FILE_PATH)
long_rows, wide_rows = [], []

for c in contractions:
    cn = c["n"]
    e = entry(cn)
    t_tq = e["torque_onset"]

    base = {
        "archivo": archivo,
        "contraccion": cn,
        "repeticion": c["repeticion"],
        "movimiento": c["movimiento"],
        "mov_id": c["mov_id"],
        "mov_inicio_s": c["start_time"],
        "mov_fin_s": c["end_time"],
    }
    wide = {
        **base,
        "torque_onset_s": t_tq,
        "torque_onset_vs_mov_inicio_ms": (t_tq - c["start_time"]) * 1000,
    }

    for k in ch_keys:
        ch = e["ch"][k]
        if ch["no_burst"]:
            onset = offset = dur = lat = lat_mov = rms = np.nan
        else:
            onset, offset = ch["onset"], ch["offset"]
            i_on, i_off = nearest_idx(onset), nearest_idx(offset)
            dur = (time[i_off] - time[i_on]) * 1000
            lat = (onset - t_tq) * 1000
            lat_mov = (onset - c["start_time"]) * 1000
            rms = float(np.sqrt(np.mean(emg[k][i_on:i_off + 1] ** 2)))

        long_rows.append({
            **base,
            "canal": k,
            "columna_csv": EMG_CHANNELS[k],
            "sin_burst": ch["no_burst"],
            "torque_onset_s": t_tq,
            "emg_onset_s": onset,
            "emg_offset_s": offset,
            "burst_duracion_ms": dur,
            "latencia_emg_vs_torque_ms": lat,
            "emg_onset_vs_mov_inicio_ms": lat_mov,
            "rms_burst_uV": rms,
        })
        wide.update({
            f"{k}_onset_s": onset,
            f"{k}_offset_s": offset,
            f"{k}_dur_ms": dur,
            f"{k}_lat_vs_torque_ms": lat,
            f"{k}_rms_uV": rms,
        })

    wide_rows.append(wide)

burst_df = pd.DataFrame(long_rows)
summary_df = pd.DataFrame(wide_rows)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("\n" + "=" * 60 + "\nBURST POR CONTRACCIÓN Y CANAL\n" + "=" * 60)
display(burst_df)
print("\n" + "=" * 60 + "\nRESUMEN POR CONTRACCIÓN (formato ancho)\n" + "=" * 60)
display(summary_df)


# %% Guardar resultados

os.makedirs(OUTPUT_DIR, exist_ok=True)


def upsert_csv(new_df, path):
    """Reemplaza las filas previas del mismo archivo (evita duplicados al re-ejecutar)."""
    if os.path.exists(path):
        old = pd.read_csv(path, encoding="utf-8-sig")
        if "archivo" in old.columns:
            old = old[old["archivo"] != new_df["archivo"].iloc[0]]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    try:
        out.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        raise PermissionError(f"No se pudo escribir {path}. ¿Está abierto en Excel?")


upsert_csv(burst_df, BURST_OUTPUT_CSV)
upsert_csv(summary_df, SUMMARY_OUTPUT_CSV)

print("✓ RESULTADOS GUARDADOS")
print("\nBurst por canal:\n ", BURST_OUTPUT_CSV)
print("\nResumen por contracción:\n ", SUMMARY_OUTPUT_CSV)


# %% (Opcional) Figuras de revisión: torque + los 4 canales apilados, una por contracción
#
# Esto NO es para marcar: es una verificación visual estática, con tamaño
# libre (se guarda a PNG), de las marcas ya hechas.

if MAKE_REVIEW_PLOTS:
    os.makedirs(REVIEW_DIR, exist_ok=True)
    for c in contractions:
        cn = c["n"]
        e = entry(cn)
        t0 = max(time[0], c["start_time"] - PRE_MARGIN_S)
        t1 = min(time[-1], c["end_time"] + POST_MARGIN_S)
        sl = slice(int(np.searchsorted(time, t0)), int(np.searchsorted(time, t1)) + 1)

        fig_r, axs = plt.subplots(1 + n_ch, 1, sharex=True, figsize=(11, 2.0 + 1.7 * (1 + n_ch)))
        axs[0].plot(time[sl], torque[sl], color="tab:blue", lw=1.1)
        axs[0].set_ylabel("Torque (N·m)")
        axs[0].set_title(f"Contracción {cn} · Rep {c['repeticion']} · {c['movimiento']} · mov #{c['mov_id']}")

        for ax, k in zip(axs[1:], ch_keys):
            ax.plot(time[sl], emg[k][sl], color=CH_COLORS[k], lw=0.5)
            ax.set_ylabel(f"{k} (µV)")
            ch = e["ch"][k]
            if ch["no_burst"]:
                ax.text(0.01, 0.9, "sin burst", transform=ax.transAxes, va="top", fontsize=8)
            elif ch["onset"] is not None and ch["offset"] is not None:
                ax.axvspan(ch["onset"], ch["offset"], color="tab:green", alpha=0.15)
                ax.axvline(ch["onset"], color="tab:green", lw=1.6)
                ax.axvline(ch["offset"], color="darkorange", lw=1.6)

        for ax in axs:
            ax.axvline(c["start_time"], color="0.45", ls=":", lw=1.2)
            ax.axvline(c["end_time"], color="0.45", ls=":", lw=1.2)
            ax.axvline(e["torque_onset"], color="red", ls="--", lw=1.6)
            ax.grid(alpha=0.3)
        axs[-1].set_xlabel("Tiempo (s)")
        axs[-1].set_xlim(t0, t1)

        fig_r.tight_layout()
        fig_r.savefig(os.path.join(REVIEW_DIR, f"revision_contr_{cn:02d}.png"), dpi=130)
        plt.close(fig_r)

    print(f"✓ Figuras de revisión en: {REVIEW_DIR}")