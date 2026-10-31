"""Reference frames, tilt and sensor fusion.

The node measures in a frame fixed to the device, not to the body. These
functions show what changes in the six channels when the device is mounted
differently, and what can be recovered.

Conventions:

* a window is (N, 6): columns 0:3 accelerometer [m/s^2], columns 3:6
  gyroscope [rad/s], as in config.SENSOR_CHANNELS;
* device frame as on Android: x to the right of the screen, y to the top of
  the screen, z out of the screen;
* at rest with the screen up the accelerometer reads (0, 0, +g): it measures
  specific force, the reaction to the support;
* angles in radians; roll about x, pitch about y, yaw about z; ZYX
  composition, R = Rz Ry Rx.

Angular velocity is a pseudovector. Under a proper rotation (det R = +1) it
transforms like a vector, so the same R applies to both sensors. Under a
reflection (det R = -1) it does not, and rotate refuses such matrices.
"""

from __future__ import annotations

import numpy as np

from iomtlab.config import TARGET_HZ

__all__ = ["GRAVITY", "rotation_matrix", "rotate", "tilt_angles", "complementary_filter"]

#: Gravity used in synthetic signals and checks, m/s^2.
GRAVITY = 9.81


# ─────────────────────────────────────────────────── rotation matrices
def rotation_matrix(roll: float = 0.0, pitch: float = 0.0,
                    yaw: float = 0.0) -> np.ndarray:
    """ZYX rotation matrix from Euler angles in radians: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)   # rotazione attorno a x
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)   # attorno a y
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)   # attorno a z
    # compongo nell'ordine ZYX: applico prima il roll, poi il pitch, poi lo yaw
    return rz @ ry @ rx


# ─────────────────────────────────────────────────── orientation tools
def rotate(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply the rotation R (3, 3) to the window X (N, 6).

    Accelerometer and gyroscope rotate with the same matrix. Returns (N, 6)
    with the dtype of X. Raises ValueError if R is not a 3x3 orthogonal
    matrix with det R = +1.
    """
    X = np.asarray(X)
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R deve essere (3, 3), non {R.shape}")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):    # controllo che R sia ortogonale
        raise ValueError("R non è ortogonale")
    if np.linalg.det(R) < 0:                               # e che non sia una riflessione
        raise ValueError("R è una riflessione (det < 0): il giroscopio non si "
                         "trasforma come un vettore sotto riflessione")
    out = np.empty_like(X, dtype=np.float64)
    # i campioni sono righe, quindi moltiplico a destra per la trasposta di R
    out[:, 0:3] = X[:, 0:3] @ R.T                  # ruoto l'accelerometro
    out[:, 3:6] = X[:, 3:6] @ R.T                  # e il giroscopio con la stessa matrice
    return out.astype(X.dtype, copy=False)


def tilt_angles(acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll and pitch from the direction of gravity, sample by sample.

    Valid when the dynamic acceleration is small compared with g, hence the
    low-pass first (lowpass_ma). Yaw is not observable from an accelerometer:
    a rotation about the vertical leaves gravity unchanged.

    acc is (N, 3) or (N, 6), in which case the first three columns are used.
    Returns (roll, pitch), each (N,) in radians.
    """
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] not in (3, 6):
        raise ValueError(f"acc deve essere (N, 3) o (N, 6), non {a.shape}")
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    roll = np.arctan2(ay, az)                      # roll dall'angolo della gravità nel piano y-z
    pitch = np.arctan2(-ax, np.hypot(ay, az))      # pitch da x contro la norma di y e z
    return roll, pitch


def complementary_filter(X: np.ndarray, fs: float = TARGET_HZ,
                         tau: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the accelerometer tilt with the integrated gyroscope.

    A recursive estimator with one parameter::

        alpha  = tau / (tau + dt)
        ang[k] = alpha * (ang[k-1] + omega[k] * dt) + (1 - alpha) * ang_acc[k]

    The gyroscope gives the high frequencies (and a slow drift), the
    accelerometer the low ones (and motion noise); tau [s] sets the
    crossover. tau -> 0 leaves the accelerometer alone, tau -> inf the pure
    integration, which drifts. The initial state is the accelerometer tilt of
    the first sample.

    X is (N, 6) sampled at fs [Hz]. Returns (roll, pitch), each (N,) in radians.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(f"X deve essere (N, 6), non {X.shape}")
    dt = 1.0 / float(fs)
    alpha = tau / (tau + dt)                 # il peso del giroscopio: vicino a 1 se tau è molto più grande di dt
    roll_a, pitch_a = tilt_angles(X)         # l'inclinazione dal solo accelerometro, campione per campione
    gx, gy = X[:, 3], X[:, 4]                      # velocità angolari attorno a x e y

    roll = np.empty(len(X))
    pitch = np.empty(len(X))
    roll[0], pitch[0] = roll_a[0], pitch_a[0]   # parto dall'inclinazione dell'accelerometro, non da zero
    for k in range(1, len(X)):
        # integro il giroscopio sul passo dt e correggo verso l'accelerometro con peso 1 - alpha
        roll[k] = alpha * (roll[k - 1] + gx[k] * dt) + (1 - alpha) * roll_a[k]
        pitch[k] = alpha * (pitch[k - 1] + gy[k] * dt) + (1 - alpha) * pitch_a[k]
    return roll, pitch
