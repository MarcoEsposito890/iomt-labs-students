"""Property checks for library functions, used by test_library_properties.py.

Each check takes the function under test, runs it on a synthetic case whose
answer is known exactly, and raises AssertionError at the first violated
property; the message names the property. They were the checks of the
exercises of earlier lab formats. The labs no longer ask students to write
these functions, so the checks now test the library.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from iomtlab.config import N_CLASSES, SENSOR_CHANNELS

__all__ = [
    "framing", "pack_window", "remaining_length",
    "phase_table", "delivery_regularity", "align_to_common_grid", "label_window",
    "timing_report", "device_offset",
    "resample_to_hz", "window_properties", "rfft_magnitude", "fft_cost",
    "make_windows", "lowpass_ma", "rotate", "tilt_angles",
    "complementary_filter",
    "class_weights", "train_epoch",
]


def _ok(name: str, n: int) -> None:
    print(f"{name}: {n} verifiche superate")


def _check(cond: bool, msg: str) -> int:
    if not cond:
        raise AssertionError(msg)
    return 1


def _close(a, b, tol, msg) -> int:
    return _check(abs(float(a) - float(b)) <= tol, f"{msg} (ottenuto {float(a):.6g})")


def _frame(t_ms, X: np.ndarray | None = None, labels=None) -> pd.DataFrame:
    """A recording with timestamps t_ms (ms), channels X (n, 6) and labels."""
    t_ms = np.asarray(t_ms, dtype=np.float64)
    n = len(t_ms)
    X = np.zeros((n, 6)) if X is None else X
    d = {"timestamp": t_ms.astype(np.int64) if np.allclose(t_ms, np.round(t_ms))
         else t_ms}
    d.update({c: X[:, i] for i, c in enumerate(SENSOR_CHANNELS)})
    d["label"] = labels if labels is not None else ["SEATED"] * n
    return pd.DataFrame(d)


# ─────────────────────────────────────────────────── framing and payload (Lab 1)
class _Trickle:
    """A fake socket: recv returns at most `step` bytes of the stream."""

    def __init__(self, data: bytes, step: int):
        self.data, self.step = data, step

    def recv(self, n: int) -> bytes:
        k = min(n, self.step)
        out, self.data = self.data[:k], self.data[k:]
        return out


def framing(frame, read_frame) -> None:
    """``frame(payload) -> bytes`` and ``read_frame(sock) -> bytes | None``."""
    import struct
    k = 0
    msgs = [b"", b"a", bytes(range(256)) * 3, b"\n" * 10, b"\x00\x01"]
    for m in msgs:
        f = frame(m)
        k += _check(len(f) == len(m) + 2, "il prefisso di lunghezza e' di 2 byte")
        k += _check(struct.unpack("!H", f[:2])[0] == len(m),
                    "i primi due byte sono la lunghezza del payload, big-endian")
    try:
        frame(bytes(70_000))
    except ValueError:
        k += 1
    else:
        raise AssertionError("70 000 byte non stanno in 2 byte di lunghezza: "
                             "frame deve sollevare ValueError")
    stream = b"".join(frame(m) for m in msgs)
    for step in (1, 3, 7, len(stream)):
        sock = _Trickle(stream, step)
        got = [read_frame(sock) for _ in msgs]
        k += _check([bytes(g) for g in got] == msgs,
                    f"con recv che restituisce al massimo {step} byte i messaggi "
                    f"tornano diversi: recv(n) puo' restituire meno di n byte")
        k += _check(read_frame(sock) is None,
                    "a fine flusso, sul confine di un frame, read_frame restituisce None")
    try:
        read_frame(_Trickle(frame(b"0123456789")[:7], 4))
    except ConnectionError:
        k += 1
    else:
        raise AssertionError("un flusso che si chiude dentro un frame deve sollevare "
                             "ConnectionError")
    _ok("framing", k)


def pack_window(fn) -> None:
    """``fn(seq, t0_ms, X) -> bytes`` in the format of iomtlab.net.payload."""
    import struct
    from iomtlab.net.payload import lsb, unpack_window
    k = 0
    step = lsb("sp")
    X = np.random.default_rng(1).normal(0, 1, (50, 6)) * np.array([3, 3, 3, 1, 1, 1])
    b = fn(7, 123456, X)
    k += _check(len(b) == 11 + 2 * 50 * 6, "11 byte di intestazione e 2 per campione")
    k += _check(struct.unpack("!IIHB", b[:11]) == (7, 123456, 50, 6),
                "intestazione: seq, t0_ms, n, ch come uint32 uint32 uint16 uint8, big-endian")
    _, _, Y = unpack_window(bytes(b))
    k += _check((np.abs(Y - X) / step).max() <= 0.5 + 1e-9,
                "arrotondando al codice piu' vicino l'errore e' al massimo mezzo passo")
    Z = np.zeros((50, 6))
    Z[1, 0] = 1000 * step[0]
    k += _check(struct.unpack("!h", bytes(fn(0, 0, Z)[23:25]))[0] == 1000,
                "i campioni vanno per campione: il campione 1 del canale 0 sta al byte 23")
    S = np.zeros((50, 6))
    S[0, 0], S[0, 1] = 2 * 32767 * step[0], -2 * 32767 * step[0]
    _, _, Ys = unpack_window(bytes(fn(0, 0, S)))
    k += _check(Ys[0, 0] > 0 and Ys[0, 1] < 0,
                "oltre il fondo scala il valore satura, non cambia segno")
    k += _close(Ys[0, 0], 32767 * step[0], 1e-9, "saturazione positiva a +32767")
    _ok("pack_window", k)


def remaining_length(fn) -> None:
    """``fn(buf, i) -> (value, n_bytes)``, every length of the MQTT table."""
    from iomtlab.net.mqtt_wire import encode_remaining_length
    k = 0
    for x in (0, 1, 127, 128, 321, 16_383, 16_384, 2_097_151, 2_097_152, 268_435_455):
        enc = encode_remaining_length(x)
        k += _check(tuple(fn(b"\x30" + enc + b"\xff" * 4, 1)) == (x, len(enc)),
                    f"{enc.hex(' ')} codifica {x} in {len(enc)} byte")
        k += _check(tuple(fn(b"\x00" * 5 + enc, 5)) == (x, len(enc)),
                    "la decodifica parte dall'indice i, non dall'inizio del buffer")
    for bad in (b"\x30\x80", b"\x30\xff\xff\xff\xff\x01"):
        try:
            fn(bad, 1)
        except ValueError:
            k += 1
        else:
            raise AssertionError(f"{bad.hex(' ')}: deve sollevare ValueError")
    _ok("remaining_length", k)


# ─────────────────────────────────────────────────── one trial, two devices (Lab 2)
def phase_table(fn) -> None:
    """``fn(df) -> DataFrame`` with one row per contiguous phase."""
    k = 0
    seq = (["SEATED"] * 5 + ["STANDING_UP"] * 3 + ["WALKING"] * 6 + ["TURNING"] * 2
           + ["WALKING"] * 6 + ["TURNING"] * 2 + ["SITTING_DOWN"] * 3 + ["SEATED"] * 4)
    t = np.arange(len(seq)) * 10 + 1_000_000
    out = fn(pd.DataFrame({"timestamp": t, "label": seq}))
    k += _check(list(out["label"]) == ["SEATED", "STANDING_UP", "WALKING", "TURNING",
                                       "WALKING", "TURNING", "SITTING_DOWN", "SEATED"],
                "otto fasi nell'ordine della prova: le due camminate e le due svolte "
                "restano separate")
    k += _close(out["t0"].iloc[2], 1_000_000 + 80, 0, "t0 della prima camminata")
    k += _close(out["dur_s"].iloc[2], 0.05, 1e-9,
                "durata della prima camminata, dal primo all'ultimo campione, in s")
    _ok("phase_table", k)


def delivery_regularity(fn) -> None:
    """``fn(df_sp, df_sw) -> dict``: same mean period, one device in bursts."""
    k = 0
    rng = np.random.default_rng(0)
    N = 3000
    t_sw = np.cumsum(np.round(np.clip(rng.normal(10.0, 2.0, N), 0, None)))
    # telefono: stesso periodo medio, consegnato a gruppi di tre campioni
    t_sp = np.repeat(np.arange(0, N // 3) * 30.0, 3)[:N]
    t_sp = t_sp + np.repeat(rng.normal(0, 1.0, N // 3 + 1), 3)[:N].round()
    t_sp = np.maximum.accumulate(np.round(t_sp))
    out = fn(_frame(t_sp), _frame(t_sw))
    k += _check(float(out["std_ratio"]) > 2.0,
                "la consegna a gruppi deve risultare molto piu' dispersa")
    k += _check(float(out["duplicate_ratio"]) > 2.0,
                "il telefono deve risultare con molti piu' timestamp duplicati")
    rev = fn(_frame(t_sw), _frame(t_sp))
    k += _close(float(rev["std_ratio"]) * float(out["std_ratio"]), 1.0, 0.05,
                "scambiando i due dispositivi il rapporto si inverte")
    same = fn(_frame(t_sw), _frame(t_sw))
    k += _close(float(same["std_ratio"]), 1.0, 1e-6, "su due registrazioni uguali il rapporto vale 1")
    _ok("delivery_regularity", k)


def align_to_common_grid(fn) -> None:
    """``fn(df_a, df_b) -> (a, b)`` on the intersection of the two time ranges."""
    k = 0
    rng = np.random.default_rng(0)
    a = _frame(np.arange(0, 10_000, 20.0), rng.normal(size=(500, 6)))
    b = _frame(np.arange(3_000, 13_000, 20.0), rng.normal(size=(500, 6)))
    ra, rb = fn(a, b)
    k += _check(len(ra) == len(rb), "le due registrazioni allineate hanno la stessa lunghezza")
    t = ra["timestamp"].to_numpy()
    k += _check(t.min() >= 3_000 - 25 and t.max() <= 10_000 + 25,
                "la griglia copre l'intersezione dei due intervalli")
    c = _frame(np.arange(50_000, 60_000, 20.0), rng.normal(size=(500, 6)))
    try:
        fn(a, c)
    except ValueError:
        k += 1
    else:
        raise AssertionError("su due registrazioni che non si sovrappongono deve "
                             "sollevare ValueError")
    _ok("align_to_common_grid", k)


def label_window(fn) -> None:
    """``fn(y_seg) -> (label, pure)``: purity and the exact tie."""
    k = 0
    lab, pure = fn(np.zeros(50, dtype=np.int64))
    k += _check(int(lab) == 0 and bool(pure), "una finestra di una sola classe e' pura")
    half = np.array([0] * 25 + [3] * 25, dtype=np.int64)
    lab, pure = fn(half)
    k += _check(not pure, "una finestra meta' e meta' e' ambigua")
    k += _check(int(fn(half[::-1].copy())[0]) == int(lab),
                "in caso di pareggio la classe scelta non dipende dall'ordine dei campioni")
    lab3, pure3 = fn(np.array([2] * 45 + [4] * 5, dtype=np.int64))
    k += _check(int(lab3) == 2 and bool(pure3), "al 90 % di una classe la finestra e' pura")
    _ok("label_window", k)


# ─────────────────────────────────────────────────── timing
def timing_report(fn) -> None:
    """``fn(df) -> TimingReport``."""
    k = 0
    t = np.arange(0, 2001, 10.0)                       # 201 samples, 10 ms apart
    r = fn(_frame(t))
    k += _check(r.n_samples == 201, "n_samples deve contare i campioni, non gli intervalli")
    k += _close(r.duration_s, 2.0, 1e-9, "duration_s: ultimo meno primo timestamp, in secondi")
    k += _close(r.mean_hz, 100.0, 1e-6, "mean_hz: intervalli divisi per durata")
    k += _close(r.median_dt_ms, 10.0, 1e-9, "median_dt_ms su griglia regolare")
    k += _close(r.std_dt_ms, 0.0, 1e-9, "std_dt_ms deve essere nulla su griglia regolare")
    k += _check(r.n_gaps == 0, "nessun buco su griglia regolare")

    t2 = t.copy()
    t2[100:] += 40.0                                   # one 50 ms interval
    t2[50] = t2[49]                                    # one duplicated timestamp
    r = fn(_frame(t2))
    k += _close(r.max_dt_ms, 50.0, 1e-9, "max_dt_ms deve vedere il buco")
    k += _check(r.n_gaps == 1, "un intervallo > 3x mediana e' esattamente un buco")
    dt = np.diff(t2)
    k += _close(r.std_dt_ms, dt.std(), 1e-9,
                "std_dt_ms: deviazione di popolazione, duplicati inclusi")
    k += _close(r.p99_dt_ms, np.percentile(dt, 99), 1e-9, "p99_dt_ms: 99-esimo percentile")
    _ok("timing_report", k)


def device_offset(fn) -> None:
    """``fn(df_sp, df_sw) -> dict``."""
    k = 0
    sp = _frame(np.arange(1000, 11001, 10.0))
    sw = _frame(np.arange(1250, 11501, 25.0))
    d = fn(sp, sw)
    k += _close(d["start_offset_ms"], 250.0, 1e-9, "start_offset_ms: orologio minus telefono")
    k += _close(d["end_offset_ms"], 500.0, 1e-9, "end_offset_ms: orologio minus telefono")
    k += _close(d["duration_sp_s"], 10.0, 1e-9, "duration_sp_s in secondi")
    k += _close(d["duration_sw_s"], 10.25, 1e-9, "duration_sw_s in secondi")
    k += _check(d["n_samples_sp"] == len(sp) and d["n_samples_sw"] == len(sw),
                "numero di campioni per dispositivo")
    _ok("device_offset", k)


# ─────────────────────────────────────────────────── sampling and spectrum
def resample_to_hz(fn) -> None:
    """``fn(df, target_hz) -> (t, X, y)``."""
    k = 0
    rng = np.random.default_rng(0)
    dt = rng.uniform(6.0, 14.0, 400)                   # about 100 Hz, with jitter
    t_ms = 5000.0 + np.concatenate([[0.0], np.cumsum(dt)])
    ts = (t_ms - t_ms[0]) / 1000.0
    X = np.stack([2.0 * ts + c for c in range(6)], axis=1)   # linear, so interpolation is exact
    labels = np.where(ts < 1.0, "SEATED", "WALKING")
    t, Xn, y = fn(_frame(t_ms, X, list(labels)), 50.0)

    k += _close(t[0], 0.0, 1e-12, "la griglia parte da zero, non dal timestamp UNIX")
    k += _check(np.allclose(np.diff(t), 1 / 50.0), "la griglia deve essere uniforme a 1/fs")
    k += _check(t[-1] <= ts[-1] + 1e-12, "nessun istante oltre l'ultimo campione: sarebbe estrapolazione")
    k += _check(Xn.shape == (len(t), 6) and len(y) == len(t), "forme (N,), (N, 6), (N,)")
    k += _check(Xn.dtype == np.float32, "i canali vanno restituiti in float32")
    k += _check(np.allclose(Xn[:, 3], 2.0 * t + 3, atol=1e-4),
                "un segnale lineare deve essere ricostruito esattamente dall'interpolazione lineare")
    k += _check(np.all(y[t < 1.0] == "SEATED") and np.all(y[t >= 1.0 + 0.015] == "WALKING"),
                "etichette: si prende l'ultimo campione grezzo non successivo all'istante, "
                "non si interpolano")
    k += _check(set(np.unique(y)) <= {"SEATED", "WALKING"},
                "un'etichetta non puo' assumere valori che non erano nei dati")

    # a 3 Hz sine: small interpolation error
    f0 = 3.0
    Xs = np.tile(np.sin(2 * np.pi * f0 * ts)[:, None], (1, 6))
    t, Xn, _ = fn(_frame(t_ms, Xs), 50.0)
    err = np.max(np.abs(Xn[:, 0] - np.sin(2 * np.pi * f0 * t)))
    k += _check(err < 0.03, f"errore di interpolazione troppo grande su 3 Hz: {err:.3f}")
    _ok("resample_to_hz", k)


# ENBW [bins], scalloping loss [dB], coherent gain.
# Closed-form values for the numpy windows; they match Harris (1978), Table 1,
# up to rounding. For Hamming the computation gives 1.75 dB and the tolerance
# covers the tabulated value.
_HARRIS = {
    "rect": (1.00, 3.92, 1.00),
    "hann": (1.50, 1.42, 0.50),
    "hamming": (1.36, 1.75, 0.54),
    "blackman": (1.73, 1.10, 0.42),
}


def window_properties(fn) -> None:
    """``fn(name, n) -> WindowProperties``, compared with Harris (1978)."""
    k = 0
    for name, (enbw, sl, cg) in _HARRIS.items():
        r = fn(name, 1024)
        k += _close(r.enbw, enbw, 0.02, f"ENBW di {name}")
        k += _close(r.scalloping_loss_db, sl, 0.06, f"scalloping loss di {name}")
        k += _close(r.coherent_gain, cg, 0.01, f"guadagno coerente di {name}")
    _ok("window_properties", k)


def rfft_magnitude(fn) -> None:
    """``fn(x, fs, window, nfft=None) -> (f, mag)``."""
    k = 0
    fs, n = 50.0, 250
    t = np.arange(n) / fs
    for w in ("rect", "hann"):
        f, m = fn(2.0 * np.sin(2 * np.pi * 5.0 * t) + 0.7, fs, w)
        k += _check(len(f) == len(m) == n // 2 + 1, "asse di rfft: n//2 + 1 bin")
        k += _close(f[1] - f[0], fs / n, 1e-12, "spaziatura dei bin fs/n")
        k += _close(f[np.argmax(m)], 5.0, 1e-9, "il picco deve cadere a 5 Hz")
        k += _close(m.max(), 2.0, 0.04, f"ampiezza su bin con finestra {w}: "
                    "manca il fattore 2/n o la correzione del guadagno coerente")
        k += _check(m[0] < 0.05, "la componente continua va rimossa prima della finestra")

    x = np.sin(2 * np.pi * 5.0 * t[:50]) + np.sin(2 * np.pi * 5.4 * t[:50])
    _, m8 = fn(x, fs, "hann", nfft=400)
    peaks = np.sum((m8[1:-1] > m8[:-2]) & (m8[1:-1] > m8[2:]) & (m8[1:-1] > 0.5 * m8.max()))
    k += _check(peaks == 1, "con 1 s di segnale 5.0 e 5.4 Hz non si separano, "
                "neanche con zero padding")
    f8, _ = fn(x, fs, "hann", nfft=400)
    k += _close(f8[1] - f8[0], fs / 400, 1e-12, "lo zero padding infittisce l'asse")
    _ok("rfft_magnitude", k)


def fft_cost(fn) -> None:
    """``fn(n, real_input=False) -> dict``."""
    k = 0
    for n in (64, 256, 1024):
        c = fn(n)
        k += _check(c["butterflies"] == (n // 2) * int(math.log2(n)),
                    "farfalle: (N/2) log2 N")
        k += _check(c["total_ops"] == 10 * c["butterflies"],
                    "una farfalla costa 4 moltiplicazioni e 6 somme reali")
    k += _check(fn(64)["total_ops"] == 1920, "N = 64: 1 920 operazioni")
    k += _check(fn(256, real_input=True)["total_ops"] == fn(256)["total_ops"] // 2,
                "ingresso reale: costo dimezzato")
    try:
        fn(100)
    except ValueError:
        k += 1
    else:
        raise AssertionError("N non potenza di due: deve sollevare ValueError")
    _ok("fft_cost", k)


# ─────────────────────────────────────────────────── gravity and rotations
def lowpass_ma(fn) -> None:
    """``fn(X, cutoff_hz, fs) -> X_filtered``."""
    k = 0
    fs, n = 50.0, 500
    t = np.arange(n) / fs
    const = np.full((n, 6), 9.81, dtype=np.float32)
    out = fn(const, 1.0, fs)
    k += _check(out.shape == const.shape, "stessa forma dell'ingresso")
    k += _check(np.allclose(out[60:-60], 9.81, atol=1e-4),
                "una costante (la gravita') deve passare invariata lontano dai bordi")
    pulse = np.zeros((n, 6), dtype=np.float32)
    pulse[250] = 1.0
    out = fn(pulse, 2.0, fs)
    k += _check(int(np.argmax(out[:, 0])) == 250,
                "fase nulla: il picco non deve spostarsi")
    k += _check(np.allclose(out[200:250, 0], out[251:301, 0][::-1], atol=1e-6),
                "fase nulla: la risposta all'impulso deve essere simmetrica")
    hf = np.tile(np.sin(2 * np.pi * 10.0 * t)[:, None], (1, 6)).astype(np.float32)
    out = fn(hf, 1.0, fs)
    k += _check(np.abs(out[60:-60]).max() < 0.1, "10 Hz deve essere attenuato con taglio a 1 Hz")
    _ok("lowpass_ma", k)


def rotate(fn) -> None:
    """``fn(X, R) -> X_rotated``, X of shape (N, 6)."""
    from iomtlab.signal.rotations import rotation_matrix, GRAVITY
    k = 0
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1.0, (200, 6))
    X[:, 2] += GRAVITY

    out = fn(X, np.eye(3))
    k += _check(np.shape(out) == X.shape, "stessa forma dell'ingresso")
    k += _check(np.allclose(out, X, atol=1e-9),
                "la rotazione identica non deve cambiare nulla")

    R = rotation_matrix(roll=0.3, pitch=-0.2, yaw=1.1)
    out = np.asarray(fn(X, R), dtype=np.float64)

    for name, sl in (("accelerometro", slice(0, 3)), ("giroscopio", slice(3, 6))):
        k += _close(np.abs(np.linalg.norm(out[:, sl], axis=1)
                           - np.linalg.norm(X[:, sl], axis=1)).max(), 0.0, 1e-8,
                    f"una rotazione conserva il modulo: {name}")

    k += _check(not np.allclose(out[:, 3:6], X[:, 3:6], atol=1e-6),
                "anche il GIROSCOPIO deve ruotare: R e' una rotazione propria, "
                "quindi la velocita' angolare si trasforma come un vettore")

    ang_before = np.einsum("ij,ij->i", X[:, 0:3], X[:, 3:6])
    ang_after = np.einsum("ij,ij->i", out[:, 0:3], out[:, 3:6])
    k += _close(np.abs(ang_after - ang_before).max(), 0.0, 1e-8,
                "l'angolo fra le due terne non puo' cambiare: se cambia, le due "
                "terne non sono state ruotate con la stessa matrice")

    R1 = rotation_matrix(roll=0.4)
    R2 = rotation_matrix(pitch=0.25)
    k += _check(np.allclose(np.asarray(fn(fn(X, R1), R2), dtype=np.float64),
                            np.asarray(fn(X, R2 @ R1), dtype=np.float64), atol=1e-8),
                "comporre due rotazioni deve dare lo stesso risultato di applicarle "
                "una dopo l'altra: attenzione all'ordine e alla trasposta")

    back = np.asarray(fn(fn(X, R), R.T), dtype=np.float64)
    k += _check(np.allclose(back, X, atol=1e-8),
                "applicare R e poi la sua trasposta deve riportare all'originale")
    _ok("rotate", k)


def tilt_angles(fn) -> None:
    """``fn(acc) -> (roll, pitch)``, in radians."""
    from iomtlab.signal.rotations import rotation_matrix, GRAVITY
    k = 0
    n = 50
    flat = np.tile([0.0, 0.0, GRAVITY], (n, 1))
    roll, pitch = fn(flat)
    k += _check(np.shape(roll) == (n,) and np.shape(pitch) == (n,),
                "roll e pitch devono avere un valore per campione")
    k += _close(np.abs(roll).max(), 0.0, 1e-6, "schermo in alto: roll nullo")
    k += _close(np.abs(pitch).max(), 0.0, 1e-6, "schermo in alto: pitch nullo")

    for r_deg, p_deg in ((30.0, 0.0), (0.0, -20.0), (-15.0, 25.0)):
        R = rotation_matrix(roll=np.deg2rad(r_deg), pitch=np.deg2rad(p_deg))
        a = np.tile((R.T @ np.array([0.0, 0.0, GRAVITY])), (n, 1))
        rr, pp = fn(a)
        k += _close(np.rad2deg(rr[0]), r_deg, 1e-3, f"roll atteso {r_deg} gradi")
        k += _close(np.rad2deg(pp[0]), p_deg, 1e-3, f"pitch atteso {p_deg} gradi")

    yaw = rotation_matrix(yaw=1.0)
    a_yaw = np.tile((yaw.T @ np.array([0.0, 0.0, GRAVITY])), (n, 1))
    rr, pp = fn(a_yaw)
    k += _close(np.abs(rr).max() + np.abs(pp).max(), 0.0, 1e-6,
                "lo yaw non e' osservabile dall'accelerometro: ruotare attorno "
                "alla verticale non deve cambiare roll ne' pitch")

    scaled = fn(flat * 0.5)
    k += _close(np.abs(scaled[0]).max(), 0.0, 1e-6,
                "gli angoli dipendono dalla direzione, non dal modulo: serve atan2, "
                "non arcsin(a / 9.81)")
    _ok("tilt_angles", k)


def complementary_filter(fn) -> None:
    """``fn(X, fs, tau) -> (roll, pitch)``."""
    from iomtlab.signal.rotations import GRAVITY
    k = 0
    fs, n = 50.0, 1500
    dt = 1.0 / fs

    rate = np.deg2rad(3.0)                      # constant rotation about x
    ang0 = np.deg2rad(20.0)                     # starting away from zero
    ang = ang0 + rate * np.arange(n) * dt       # stays within +-180, no wrap
    X = np.zeros((n, 6))
    X[:, 0] = 0.0
    X[:, 1] = GRAVITY * np.sin(ang)
    X[:, 2] = GRAVITY * np.cos(ang)
    X[:, 3] = rate

    roll, pitch = fn(X, fs, 1.0)
    k += _check(np.shape(roll) == (n,), "un valore per campione")
    k += _close(np.rad2deg(np.abs(roll[200:] - ang[200:]).max()), 0.0, 1.0,
                "senza rumore ne' bias il filtro deve seguire la rotazione vera "
                "entro un grado")

    bias = np.deg2rad(2.0)
    Xb = X.copy(); Xb[:, 3] += bias
    roll_b, _ = fn(Xb, fs, 1.0)
    drift = float(np.rad2deg(np.abs(roll_b[-1] - ang[-1])))
    k += _check(drift < 5.0,
                "con un bias di 2 gradi/s il filtro deve restare limitato: errore "
                f"finale {drift:.1f} gradi, mentre la sola integrazione arriverebbe "
                f"a {np.rad2deg(bias * n * dt):.0f}")

    rng = np.random.default_rng(1)
    Xn = X.copy()
    Xn[:, 0:3] += rng.normal(0, 2.0, (n, 3))            # motion, not gravity
    r_acc, _ = fn(Xn, fs, 0.0)                          # tau = 0: accelerometer only
    r_mix, _ = fn(Xn, fs, 1.0)
    k += _check(np.std(np.diff(r_mix)) < np.std(np.diff(r_acc)),
                "con tau = 1 s l'uscita deve essere piu' liscia che con tau = 0, "
                "dove resta il solo accelerometro")

    k += _close(np.rad2deg(fn(X, fs, 1.0)[0][0] - ang[0]), 0.0, 1e-6,
                "lo stato iniziale deve venire dall'accelerometro, non da zero: "
                "altrimenti c'e' un transitorio lungo tau")
    _ok("complementary_filter", k)


# ─────────────────────────────────────────────────── windowing
def make_windows(fn) -> None:
    """``fn(X, y, size, shift) -> (Xw, yw, purity)``."""
    k = 0
    N, C = 203, 6
    X = np.arange(N * C, dtype=np.float32).reshape(N, C)
    y = np.zeros(N, dtype=np.int64)
    y[100:] = 2
    Xw, yw, pur = fn(X, y, 50, 25)
    nw = (N - 50) // 25 + 1
    k += _check(Xw.shape == (nw, 50, C), f"numero di finestre: (N - size)//shift + 1 = {nw}")
    k += _check(len(yw) == len(pur) == nw, "una etichetta e una purity per finestra")
    k += _check(np.array_equal(Xw[3], X[75:125]), "la finestra i parte al campione i*shift")
    k += _check(Xw.dtype == np.float32 and yw.dtype == np.int64, "dtype float32 e int64")
    k += _check(yw[0] == 0 and yw[-1] == 2, "etichetta di maggioranza")
    k += _close(pur[3], 25 / 50, 1e-6, "purity della finestra a cavallo: frazione della classe vincente")
    k += _check(yw[3] == 0, "a parita' vince l'etichetta con indice minore")
    k += _close(pur[2], 1.0, 1e-6, "purity 1 dentro un'unica classe")
    k += _check(np.all((pur > 0) & (pur <= 1)), "purity in (0, 1]")

    Xw, yw, pur = fn(X[:10], y[:10], 50, 25)
    k += _check(Xw.shape == (0, 50, C) and len(yw) == len(pur) == 0,
                "segnale piu' corto della finestra: zero finestre, forme coerenti")
    _ok("make_windows", k)


# ─────────────────────────────────────────────────── training
def class_weights(inverse_freq, effective_num) -> None:
    """``inverse_freq(y)`` and ``effective_num(y, beta=...)``."""
    k = 0
    y = np.repeat(np.arange(N_CLASSES), [400, 50, 900, 120, 60])
    w = np.asarray(inverse_freq(y), dtype=np.float64)
    counts = np.bincount(y, minlength=N_CLASSES)
    k += _check(w.shape == (N_CLASSES,), "un peso per classe")
    k += _check(np.allclose(w * counts, (w * counts)[0], rtol=1e-5),
                "inverse frequency: peso x conteggio uguale per tutte le classi")
    k += _close(w.mean(), 1.0, 1e-5, "i pesi inverse frequency hanno media 1")
    w0 = np.asarray(effective_num(y, beta=1e-9), dtype=np.float64)
    k += _check(np.allclose(w0, w0[0], rtol=1e-4), "beta -> 0: pesi uniformi")
    w1 = np.asarray(effective_num(y, beta=1 - 1e-7), dtype=np.float64)
    k += _check(np.allclose(w1 / w1.mean(), w, rtol=1e-3),
                "beta -> 1: si ritrovano i pesi inverse frequency")
    _ok("class_weights", k)


def train_epoch(fn) -> None:
    """``fn(model, loader, optimizer, criterion, device) -> float``."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    k = 0
    g = torch.Generator().manual_seed(0)
    X = torch.randn(256, 4, generator=g)
    y = (X[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.1), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    crit = nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in model.parameters()]
    model.eval()
    losses = [fn(model, loader, opt, crit, torch.device("cpu")) for _ in range(5)]
    k += _check(model.training, "la funzione deve mettere il modello in modalita' train")
    k += _check(all(isinstance(v, float) for v in losses), "restituisce un float, non un tensore")
    k += _check(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())),
                "i parametri non cambiano: manca optimizer.step()")
    k += _check(losses[-1] < 0.6 * losses[0],
                "la loss non scende: gradienti non azzerati, o backward mancante")
    _ok("train_epoch", k)
