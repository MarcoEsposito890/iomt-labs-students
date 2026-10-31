"""Compression of the example network (Lab 5): int8, pruning, distillation, four tests.

The node is the class-2 node of the theory (RFC 7228): about 250 KiB of
flash and 50 KiB of RAM, 40 MMAC/s, 0.6 nJ per MAC, an IEEE 802.15.4 radio
at 250 kbit/s and 240 nJ per bit, a 10.8 kJ battery, a 0.5 mW sensor, two
decisions per second with a deadline of 100 ms. These are declared
constants, not measurements on a node.

Quantisation is symmetric, zero point 0, one step per output channel by
default: q = round(w / s), s = max|w| / (2^(b-1) - 1). The library quantises
the weights and rebuilds the network with the rounded values, because
PyTorch has no dynamic quantisation for convolutions.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from iomtlab import radio
from iomtlab.config import ENERGY
from iomtlab.models.memory import peak_activation_values

__all__ = ["NODE", "quantize_state", "dequantize_state", "weight_error", "state_bytes",
           "int8_bytes", "update_on_link", "prune_magnitude", "shrink_example_cnn",
           "kd_loss", "train_student", "peak_activation_values", "four_tests", "quantize_array"]

#: The class-2 node of the theory. min_days and min_nodes are the requirements of
#: tests 3 and 4: the theory gives no value, so they are None unless a lab sets them.
#: listen_fraction is the share of time the radio listens (idle listening, the
#: duty cycle of the MAC): 0 means the radio is on only for its own bits, as in
#: the theory's four tests; listen_W is the power of a listening radio, 60 mW, the
#: same 60 mW that gives 240 nJ/bit at 250 kbit/s.
NODE = {
    "flash_B": 250 * 1024, "ram_B": 50 * 1024,
    "macs_per_s": 40e6, "deadline_s": 0.100,
    "e_mac_J": ENERGY["e_mac_J"], "e_bit_J": ENERGY["e_bit_lowpower_J"],
    "sensor_W": 0.5e-3, "sensor_s": 0.5,                 # 0,25 mJ di sensore per decisione
    "battery_J": 10.8e3, "decisions_per_s": 2.0,
    "rate_bps": ENERGY["rate_lowpower_bps"], "usable_fraction": ENERGY["usable_fraction"],
    "listen_fraction": 0.0, "listen_W": 60e-3,          # radio in ascolto: quota del tempo e potenza
    "min_days": None, "min_nodes": None,
}


def _weighted(model):
    """(name, module) of the layers with a weight matrix: Conv1d and Linear."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv1d, nn.Linear))]


# ══════════════════════════════════════════════════════════ quantisation
@torch.no_grad()
def quantize_state(model, bits: int = 8, per_channel: bool = True) -> dict:
    """The weights of every Conv1d and Linear layer as integers of bits bits, with their steps.

    Returns {layer: {"q": int8 tensor, "scale": float32 tensor, "bits": bits}};
    scale has one value per output channel, or one per layer if per_channel
    is False. Biases are not quantised: they stay at 32 bits.
    """
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for name, m in _weighted(model):
        w = m.weight.detach()
        if per_channel:
            mx = w.abs().reshape(w.shape[0], -1).amax(1)          # il massimo per canale di uscita
            s = mx / qmax
            shape = (-1,) + (1,) * (w.dim() - 1)
            q = torch.round(w / s.clamp_min(1e-12).reshape(shape))
        else:
            s = (w.abs().max() / qmax).reshape(1)                 # un passo solo per tutto lo strato
            q = torch.round(w / s.clamp_min(1e-12))
        out[name] = {"q": q.clamp(-qmax, qmax).to(torch.int8), "scale": s.float(), "bits": bits}
    return out


def quantize_array(w, bits: int = 8, per_channel: bool = True):
    """The rule of quantize_state on one numpy weight array w (output channels first).

    Returns (q, s): q integers in [-(2^(b-1)-1), 2^(b-1)-1] as int32, s the step,
    one per output channel (shape (out,)) or one per layer (shape (1,)), with
    zero point 0. The reference solution of the Lab 5 «Scrivi tu» exercise.
    """
    import numpy as np
    w = np.asarray(w, dtype=np.float64)
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        s = np.abs(w.reshape(w.shape[0], -1)).max(1) / qmax            # un passo per canale di uscita
        sb = np.maximum(s, 1e-12).reshape((-1,) + (1,) * (w.ndim - 1))
    else:
        s = np.array([np.abs(w).max() / qmax])                         # un passo per tutto lo strato
        sb = max(s[0], 1e-12)
    q = np.clip(np.round(w / sb), -qmax, qmax).astype(np.int32)
    return q, s


@torch.no_grad()
def dequantize_state(qs: dict, model):
    """A copy of model whose weights are the rounded values s * q of quantize_state."""
    m2 = copy.deepcopy(model)
    mods = dict(m2.named_modules())
    for name, d in qs.items():
        w = mods[name].weight
        shape = (-1,) + (1,) * (w.dim() - 1) if d["scale"].numel() > 1 else (1,)
        w.copy_(d["q"].float() * d["scale"].reshape(shape))
    return m2.eval()


@torch.no_grad()
def weight_error(model, qs: dict) -> list:
    """The rounding error of the weights, layer by layer.

    One dict per layer: layer, passo_medio (mean step s), max_err, max_su_mezzo_passo
    (the largest error divided by half the step of its channel: at most 1),
    rms and rms_su_teorico (rms divided by the theoretical s / sqrt(12)).
    """
    mods = dict(model.named_modules())
    rows = []
    for name, d in qs.items():
        w = mods[name].weight.detach()
        s = d["scale"]
        shape = (-1,) + (1,) * (w.dim() - 1) if s.numel() > 1 else (1,)
        sb = s.reshape(shape).expand_as(w) if s.numel() > 1 else s.expand_as(w)
        e = w - d["q"].float() * sb
        teor = math.sqrt(float((sb ** 2).mean()) / 12)          # s / radice di 12, mediato sui pesi
        rms = float(e.pow(2).mean().sqrt())
        rows.append({"layer": name, "passo_medio": float(s.mean()), "max_err": float(e.abs().max()),
                     "max_su_mezzo_passo": float((e.abs() / (sb / 2).clamp_min(1e-12)).max()),
                     "rms": rms, "rms_su_teorico": rms / max(teor, 1e-12)})
    return rows


def state_bytes(qs: dict) -> int:
    """Bytes of the file torch.save writes for a quantised state."""
    import io as _io
    buf = _io.BytesIO()
    torch.save(qs, buf)
    return buf.getbuffer().nbytes


def int8_bytes(model) -> dict:
    """Bytes of the model in int8, counted: weights int8, biases int32, one float32 step per channel.

    Returns weights_B, bias_B, scale_B and total_B. Layers without weights
    (pooling, ReLU) cost nothing; parameters of other layers count 4 bytes.
    """
    w = b = sc = other = 0
    counted = set()
    for _, m in _weighted(model):
        w += m.weight.numel()
        sc += 4 * m.weight.shape[0]                              # un passo per canale di uscita
        counted.add(id(m.weight))
        if m.bias is not None:
            b += 4 * m.bias.numel()
            counted.add(id(m.bias))
    for p in model.parameters():
        if id(p) not in counted:
            other += 4 * p.numel()                               # per esempio i pesi di un LSTM
    return {"weights_B": int(w), "bias_B": int(b), "scale_B": int(sc), "other_B": int(other),
            "total_B": int(w + b + sc + other)}


def update_on_link(nbytes: int, header_B: int = 0, nodes: int = 1,
                   rate_bps: float = radio.RATE_BPS) -> dict:
    """Radio time to send nbytes to one node over IEEE 802.15.4, and to nodes nodes on one channel.

    Without header_B the frames are full 127-byte frames and only the
    payload is counted (the convention of «An update on a constrained
    link»). With header_B = 17 the convention of iomtlab.radio: 116 useful
    bytes per frame and 17 bytes of headers each. Lower bounds: no
    acknowledgements, no retransmissions, no wake-up times.

    Returns bytes, frames, air_bytes, air_s (one node), nodes, fleet_s.
    """
    if header_B:
        per = radio.PSDU_B - max(0, header_B - radio.PHY_B)   # 17 B: 6 sull'aria e 11 nella trama, 116 utili
        fr = radio.frames(nbytes, per)
        air = nbytes + header_B * fr
    else:
        fr = radio.frames(nbytes, radio.PSDU_B)
        air = nbytes
    t = air * 8 / rate_bps
    return {"bytes": int(nbytes), "frames": int(fr), "air_bytes": int(air), "air_s": t,
            "nodes": nodes, "fleet_s": t * nodes}


# ══════════════════════════════════════════════════════════ pruning
@torch.no_grad()
def _masks(model, amount: float, structured: bool) -> dict:
    """{layer: mask} with the weights (or the output channels) to keep."""
    layers = _weighted(model)
    out = {}
    for i, (name, m) in enumerate(layers):
        w = m.weight.detach()
        if structured:
            if i == len(layers) - 1:
                continue                         # l'uscita ha una riga per classe: non la tocco
            l1 = w.abs().reshape(w.shape[0], -1).sum(1)        # norma L1 di ogni filtro
            k = int(round(amount * w.shape[0]))
            keep = torch.ones(w.shape[0], dtype=torch.bool)
            if k:
                keep[torch.argsort(l1)[:k]] = False            # tolgo i k filtri più piccoli
            out[name] = keep
        else:
            k = int(round(amount * w.numel()))
            mask = torch.ones_like(w, dtype=torch.bool)
            if k:
                idx = torch.argsort(w.abs().flatten())[:k]     # i k pesi più piccoli in modulo
                mask.view(-1)[idx] = False
            out[name] = mask
    return out


@torch.no_grad()
def _apply(model, masks: dict, structured: bool):
    mods = dict(model.named_modules())
    for name, mk in masks.items():
        m = mods[name]
        if structured:
            m.weight[~mk] = 0                    # il filtro intero, e con lui il suo bias
            if m.bias is not None:
                m.bias[~mk] = 0
        else:
            m.weight.mul_(mk)


def prune_magnitude(model, amount: float, structured: bool = False, X=None, y=None,
                    epochs: int = 0, seed: int = 0):
    """A pruned copy of model: in each layer the share amount of weights (or filters) set to zero.

    Unstructured: the weights with the smallest magnitude (Han et al., 2015).
    Structured: the output filters with the smallest L1 norm, with their
    bias, in every layer but the output (Li et al., 2017). With epochs > 0
    the copy is retrained on windows X (physical units) and labels y with
    the reference recipe, keeping the mask fixed after every step.
    """
    from iomtlab.train.compare import LR, make_loader
    from iomtlab.train.data import to_input
    from iomtlab.train.loops import inverse_freq_weights, set_seed
    m2 = copy.deepcopy(model)
    masks = _masks(m2, amount, structured)
    _apply(m2, masks, structured)
    if epochs > 0:
        set_seed(seed)
        loader = make_loader(to_input(X), y, seed=seed)
        loss_fn = nn.CrossEntropyLoss(weight=inverse_freq_weights(np.asarray(y)))
        opt = torch.optim.Adam(m2.parameters(), lr=LR)
        m2.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(m2(xb), yb).backward()
                opt.step()
                _apply(m2, masks, structured)    # i pesi tolti restano zero
    return m2.eval()


@torch.no_grad()
def shrink_example_cnn(model):
    """The narrower ExampleCNN equivalent to a structurally pruned one: zero filters removed.

    A filter whose weights and bias are all zero gives a zero output, so the
    filter and the column of the next layer that reads it can be removed
    without changing the logits.
    """
    from iomtlab.models import ExampleCNN

    def alive(m):
        w = m.weight.reshape(m.weight.shape[0], -1)
        return (w.abs().sum(1) > 0) | (m.bias.abs() > 0)

    k1, k2, k3, kd = (alive(m) for m in (model.conv1, model.conv2, model.conv3, model.dense))
    t = model.dense.in_features // model.conv3.out_channels          # posizioni dopo pool3
    small = ExampleCNN(in_channels=model.conv1.in_channels, num_classes=model.out.out_features,
                       width=(int(k1.sum()), int(k2.sum()), int(k3.sum())),
                       kernel=model.kernel, hidden=int(kd.sum()))
    small.conv1.weight.copy_(model.conv1.weight[k1]); small.conv1.bias.copy_(model.conv1.bias[k1])
    small.conv2.weight.copy_(model.conv2.weight[k2][:, k1]); small.conv2.bias.copy_(model.conv2.bias[k2])
    small.conv3.weight.copy_(model.conv3.weight[k3][:, k2]); small.conv3.bias.copy_(model.conv3.bias[k3])
    cols = k3.repeat_interleave(t)                    # l'appiattimento mette le t posizioni di un canale di seguito
    small.dense.weight.copy_(model.dense.weight[kd][:, cols]); small.dense.bias.copy_(model.dense.bias[kd])
    small.out.weight.copy_(model.out.weight[:, kd]); small.out.bias.copy_(model.out.bias)
    return small.eval()


# ══════════════════════════════════════════════════════════ distillation
def kd_loss(z_student, z_teacher, y, T: float = 4.0, alpha: float = 0.7, weight=None):
    """Hinton's loss: alpha T^2 KL(p_teacher^T || p_student^T) + (1 - alpha) CE(z_student, y)."""
    soft = Fn.kl_div(Fn.log_softmax(z_student / T, 1), Fn.softmax(z_teacher / T, 1),
                     reduction="batchmean") * T * T
    hard = Fn.cross_entropy(z_student, y, weight=weight)
    return alpha * soft + (1 - alpha) * hard


def train_student(student, X, y, teacher_logits=None, T: float = 4.0, alpha: float = 0.7,
                  epochs: int = 15, seed: int = 0) -> list:
    """Train student on windows X (physical units) and labels y; returns the loss of each epoch.

    With teacher_logits (one row per window of X) the loss is kd_loss;
    without, the weighted cross-entropy of the reference recipe. The batches
    depend only on seed, so a student trained with and without the teacher
    sees the same batches in the same order.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from iomtlab.train.compare import BATCH, LR
    from iomtlab.train.data import to_input
    from iomtlab.train.loops import inverse_freq_weights
    y = np.asarray(y)
    x = to_input(X)
    zt = torch.zeros(len(y), 1) if teacher_logits is None else torch.as_tensor(teacher_logits).float()
    ds = TensorDataset(x, torch.as_tensor(y, dtype=torch.long), zt)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    wts = inverse_freq_weights(y)
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    hist = []
    student.train()
    for _ in range(epochs):
        tot = 0.0
        for xb, yb, zb in loader:
            opt.zero_grad()
            z = student(xb)
            loss = (Fn.cross_entropy(z, yb, weight=wts) if teacher_logits is None
                    else kd_loss(z, zb, yb, T, alpha, wts))
            loss.backward()
            opt.step()
            tot += loss.item()
        hist.append(tot / max(1, len(loader)))
    student.eval()
    return hist


# ══════════════════════════════════════════════════════════ four tests
def four_tests(model_B: int, macs: int, act_B: int, payload_bits: int, node: dict = NODE) -> dict:
    """The four tests of a version of the model on node.

    1 fits: model_B in the flash and act_B in the RAM.
    2 fast: macs / macs_per_s within the deadline.
    3 battery: energy per decision = macs e_mac + payload_bits e_bit + sensor;
      days of battery at decisions_per_s, plus listen_fraction x listen_W of
      a radio that listens for the channel (0 by default).
    4 channel: nodes that fit in usable_fraction of the rate, each sending
      decisions_per_s messages of payload_bits in 802.15.4 frames (17 B of
      headers each).
    The energy counts the payload bits only, as the theory does; the channel
    counts the frames. Tests 3 and 4 pass when node has no min_days or
    min_nodes.
    """
    n = {**NODE, **node}
    t = macs / n["macs_per_s"]
    e_c = macs * n["e_mac_J"]
    e_r = payload_bits * n["e_bit_J"]
    e_s = n["sensor_W"] * n["sensor_s"]
    E = e_c + e_r + e_s
    p_listen = n["listen_fraction"] * n["listen_W"]          # la radio in ascolto consuma anche senza dati
    days = n["battery_J"] / (n["decisions_per_s"] * E + p_listen) / 86400
    pay_B = int(math.ceil(payload_bits / 8))
    air_bits = 8 * radio.air_bytes(pay_B)
    nodes = int(n["usable_fraction"] * n["rate_bps"] // (n["decisions_per_s"] * air_bits))
    r = {"flash_ok": model_B <= n["flash_B"], "ram_ok": act_B <= n["ram_B"],
         "t_s": t, "fast": t <= n["deadline_s"],
         "E_J": E, "E_compute_J": e_c, "E_radio_J": e_r, "E_sensor_J": e_s,
         "P_listen_W": p_listen, "days": days,
         "battery_ok": n["min_days"] is None or days >= n["min_days"],
         "frames": radio.frames(pay_B), "nodes": nodes,
         "channel_ok": n["min_nodes"] is None or nodes >= n["min_nodes"]}
    r["fits"] = r["flash_ok"] and r["ram_ok"]
    r["passes"] = r["fits"] and r["fast"] and r["battery_ok"] and r["channel_ok"]
    return r
