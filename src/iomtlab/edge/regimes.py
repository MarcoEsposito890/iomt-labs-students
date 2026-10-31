"""Where each cut wins: the rule m + z < 1, the envelope over ρh, links, constraints.

The rows are those of edge.cuts.cut_profile: the raw end first, the class last.
With e_MAC the energy per MAC on the node, e_bit the energy per bit of the
link and h the radio hops, the energy of the node at cut p divided by e_MAC is

    M_p + k b_p,        k = ρ h,  ρ = e_bit / e_MAC  (MACs that one bit costs)

so the ratio and the hops are one axis, k, in MACs per bit. Each cut is a
line in k; the cheapest cut for a given k is the lower envelope of the lines.

  rule            per cut: m, z, m + z and the interval of k where it beats
                  both ends. An interior cut can win for some k only if
                  m + z < 1 (m share of the MACs on the node, z bits at the
                  cut over the bits of the raw end).
  envelope        the intervals of k where each cut is the cheapest of all.
  regime_map      the winner over a grid of k, optionally with some options
                  excluded (a constraint): the data of a regime figure.
  Link, LINKS     a radio: power and rate, 802.15.4 frames or LoRa airtime
                  and duty cycle. LINKS holds the links of the course.
  lora_airtime_s  time on air of one LoRa packet (Semtech AN1200.13).
  link_cost       bits per decision -> frames, time on air, energy.
  head_resources  per cut: parameters and peak activation of the head.
  place           every cut with energy, delay and the tests that exclude it:
                  memory, deadline, raw data that may not leave, duty cycle.
  choose          the cheapest feasible row of place.

All energies are of the node and of the relays (h transmissions), not of the
receiver, as in edge.split.cut_cost. Constants are declared, not measured:
the energy per bit of a real radio has a fixed cost per packet (wake-up,
acknowledgement) that this model leaves out unless overhead_B carries it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["rule", "envelope", "regime_map", "Link", "LINKS", "lora", "lora_airtime_s",
           "link_cost", "head_resources", "place", "choose"]


# ── la regola e l'inviluppo: solo MAC e bit ──────────────────────────────────

def rule(rows) -> list[dict]:
    """For each row: m, z, m + z, and the interval [k_lo, k_hi] of k = ρh where it beats both ends.

    m = mac_head / total MACs, z = bits / bits of the raw end. The cut beats
    the raw end when k (b_raw - b_p) > M_p and the class when
    k (b_p - b_class) < M - M_p. Both hold for some k only if k_lo < k_hi,
    which (with the class taking ~0 bits) is m + z < 1. k_lo and k_hi are in
    MACs per bit; can_win is False for the ends and for cuts that never win.
    """
    raw, cls = rows[0], rows[-1]                     # i due estremi: grezzo e classe
    M, Z, bc = cls["mac_head"], raw["bits"], cls["bits"]
    out = []
    for i, r in enumerate(rows):
        m = r["mac_head"] / M if M else 0.0
        z = r["bits"] / Z if Z else 0.0
        end = i in (0, len(rows) - 1)
        lo = r["mac_head"] / (Z - r["bits"]) if r["bits"] < Z else math.inf
        hi = (M - r["mac_head"]) / (r["bits"] - bc) if r["bits"] > bc else math.inf
        out.append({"name": r["name"], "m": m, "z": z, "m_plus_z": m + z,
                    "k_lo": None if end else lo, "k_hi": None if end else hi,
                    "can_win": (not end) and lo < hi})
    return out


def _cost(r, k):
    return r["mac_head"] + k * r["bits"]


def envelope(rows, exclude=(), k_max: float = 1e6) -> list[dict]:
    """The intervals of k in [0, k_max] where each row is the cheapest; rows in exclude are left out.

    Returns [{name, k_from, k_to}] in increasing k; adjacent intervals of the
    same row are merged. Exact for the linear model: the breakpoints are the
    intersections of the lines M_p + k b_p.
    """
    cand = [r for r in rows if r["name"] not in set(exclude)]
    if not cand:
        raise ValueError("nessuna opzione rimasta dopo le esclusioni")
    ks = {0.0, float(k_max)}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if a["bits"] != b["bits"]:
                k = (b["mac_head"] - a["mac_head"]) / (a["bits"] - b["bits"])
                if 0 < k < k_max:
                    ks.add(float(k))
    ks = sorted(ks)
    segs = []
    for a, b in zip(ks[:-1], ks[1:]):
        mid = (a + b) / 2                            # nel mezzo dell'intervallo vince una sola retta
        w = min(cand, key=lambda r: (_cost(r, mid), r["bits"]))
        if segs and segs[-1]["name"] == w["name"]:
            segs[-1]["k_to"] = b
        else:
            segs.append({"name": w["name"], "k_from": a, "k_to": b})
    return segs


def regime_map(rows, k_values, exclusions=((),)) -> list[dict]:
    """The winner for every k in k_values and every set of excluded options.

    exclusions is a list of tuples of row names; () means nothing excluded,
    ("classe",) a node where the whole model does not fit, ("finestra",) raw
    data that may not leave. One dict per (exclusion, k): excluded, k, winner.
    """
    out = []
    for ex in exclusions:
        cand = [r for r in rows if r["name"] not in set(ex)]
        for k in k_values:
            w = min(cand, key=lambda r: (_cost(r, k), r["bits"]))
            out.append({"excluded": tuple(ex), "k": float(k), "winner": w["name"]})
    return out


# ── i collegamenti ────────────────────────────────────────────────────────────

def lora_airtime_s(payload_B: int, sf: int, bw_hz: float = 125e3, cr: int = 1,
                   preamble: int = 8, explicit_header: bool = True, crc: bool = True,
                   low_dr_opt: bool | None = None) -> float:
    """Time on air of one LoRa packet with payload_B bytes (Semtech AN1200.13).

    cr is the coding rate index (1 for 4/5); low_dr_opt defaults to on for
    SF11 and SF12 at 125 kHz. 15 B at SF12 give 1.155 s, at SF7 46.3 ms.
    payload_B is the whole PHY payload: for LoRaWAN add its 13 B of header.
    """
    de = (sf >= 11 and bw_hz <= 125e3) if low_dr_opt is None else low_dr_opt
    ts = 2 ** sf / bw_hz                             # durata di un simbolo
    h = 0 if explicit_header else 1
    n = 8 + max(math.ceil((8 * payload_B - 4 * sf + 28 + 16 * int(crc) - 20 * h)
                          / (4 * (sf - 2 * int(de)))) * (cr + 4), 0)
    return (preamble + 4.25) * ts + n * ts


@dataclass(frozen=True)
class Link:
    """A radio link: tx_W watts while transmitting at rate_bps.

    Framed links (payload_B set) carry at most payload_B useful bytes per
    frame and overhead_B bytes more on the air per frame (802.15.4: 116 and
    17). A LoRa link (lora_sf set) is timed by lora_airtime_s, frame by frame,
    with header_B bytes of LoRaWAN header in every packet; duty_cycle (0.01 in
    EU868) limits the time on air. tx_W None: energy unknown, only time.
    """
    name: str
    rate_bps: float
    tx_W: float | None
    payload_B: int | None = None
    overhead_B: int = 0
    lora_sf: int | None = None
    lora_bw_hz: float = 125e3
    header_B: int = 0
    duty_cycle: float | None = None
    source: str = ""

    @property
    def e_bit_J(self) -> float | None:
        """Energy per bit on the air, tx_W / rate_bps; None if tx_W is unknown."""
        return None if self.tx_W is None else self.tx_W / self.rate_bps


def lora(sf: int, tx_W: float | None = None, bw_hz: float = 125e3, max_payload_B: int = 51,
         header_B: int = 13, duty_cycle: float = 0.01) -> Link:
    """A LoRaWAN link at spreading factor sf, EU868 by default.

    max_payload_B is the application payload per packet (51 B at SF12 and
    SF11 in EU868, more at lower SF: check the regional parameters for yours);
    header_B the LoRaWAN header. tx_W is the transmit power drawn by your
    radio: the library has no sourced value, so the default None gives times
    and duty cycle but no energy.
    """
    rate = sf * bw_hz / 2 ** sf * 4 / 5              # bit/s nominali, CR 4/5
    return Link(f"LoRa SF{sf}", rate, tx_W, payload_B=max_payload_B, lora_sf=sf,
                lora_bw_hz=bw_hz, header_B=header_B, duty_cycle=duty_cycle,
                source="Semtech AN1200.13; LoRaWAN regional parameters EU868")


#: DENNI (Sahu et al., IPCCC 2021, §IV): 11,6 µs per byte, cioè 84,186 KiB/s; 1 J ogni 350 KiB.
_BLE_RATE = 8 / 11.6e-6                              # bit/s
_BLE_E_BIT = 1.0 / (350 * 1024 * 8)                  # J/bit, circa 0,35 µJ


def _links():
    from iomtlab import radio
    from iomtlab.config import ENERGY
    return {
        "802.15.4": Link("IEEE 802.15.4", ENERGY["rate_lowpower_bps"], 60e-3,
                         payload_B=radio.PAYLOAD_B, overhead_B=radio.OVERHEAD_B,
                         source="config.ENERGY: 60 mW a 250 kbit/s, 240 nJ/bit; trame di iomtlab.radio"),
        "mobile": Link("radio mobile", ENERGY["rate_mobile_bps"], 700e-3,
                       source="config.ENERGY: 700 mW a 10 Mbit/s, 70 nJ/bit (ipotesi del corso)"),
        "qpart": Link("radio di QPART", 200e6, 1.0,
                      source="Li et al., QPART, arXiv 2506.23934, Tabella II: 1 W a 200 Mbit/s"),
        "ble_denni": Link("BLE (modello di DENNI)", _BLE_RATE, _BLE_RATE * _BLE_E_BIT, payload_B=65,
                          source="Sahu et al., DENNI, IPCCC 2021, §IV: 11,6 µs per byte misurati (84,186 KB/s); "
                                 "1 J ogni 350 KB per PDU da 65 B, modellato da Siekkinen et al., WCNCW 2012"),
        "lora_sf7": lora(7), "lora_sf12": lora(12),
    }


#: The links of the course and of the theory; build your own Link for a measured radio.
LINKS = _links()


def link_cost(link: Link, bits: int, hops: int = 1) -> dict:
    """Frames, bits on the air, seconds of channel and joules of h transmissions for one decision.

    Unframed links send the bits as they are. Framed links round up to bytes
    and add overhead_B per frame. LoRa links add header_B per packet and time
    each packet with lora_airtime_s. energy_J is None if link.tx_W is None.
    """
    nbytes = int(math.ceil(bits / 8))
    if link.lora_sf is not None:
        n = int(math.ceil(nbytes / link.payload_B)) if nbytes else 0
        sizes = [link.payload_B] * (nbytes // link.payload_B)
        if nbytes % link.payload_B:
            sizes.append(nbytes % link.payload_B)
        t = sum(lora_airtime_s(s + link.header_B, link.lora_sf, link.lora_bw_hz) for s in sizes)
        air_bits = 8 * (nbytes + n * link.header_B)
    elif link.payload_B is not None:
        n = int(math.ceil(nbytes / link.payload_B)) if nbytes else 0
        air_bits = 8 * (nbytes + n * link.overhead_B)
        t = air_bits / link.rate_bps
    else:
        n, air_bits = (1 if bits else 0), bits
        t = bits / link.rate_bps
    e = None if link.tx_W is None else hops * link.tx_W * t
    return {"frames": n, "air_bits": air_bits, "t_air_s": hops * t, "energy_J": e}


# ── le risorse della testa ───────────────────────────────────────────────────

def head_resources(model, weight_bits: int = 8, act_bits: int = 8, rows=None) -> dict:
    """{cut name: {params, weight_B, peak_act_B}} for every boundary of model.stages().

    params counts the parameters of the modules that run up to and including
    the stage (shared modules once); weight_B = params * weight_bits / 8.
    peak_act_B is the largest input plus output of one stage up to the cut,
    at act_bits per value: a stage-level bound, coarser than
    models.memory.peak_activation_values. The raw end keeps only the window
    (16-bit samples of the active dataset); the class is the whole model.
    """
    import torch
    from iomtlab.spec import active
    S = active()
    stages = list(model.stages())
    seen, out, peak = set(), {}, 0
    used = []

    def hook(mod, inp, outp):
        used.append(mod)

    leaves = [m for m in model.modules() if not list(m.children())]
    handles = [m.register_forward_hook(hook) for m in leaves]
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, S.n_channels, S.window)
            for name, f in stages:
                n_in = int(x[0].numel())
                used.clear()
                x = f(x)
                for m in used:
                    for p in m.parameters(recurse=False):
                        seen.add(id(p))
                params = sum(p.numel() for p in model.parameters() if id(p) in seen)
                peak = max(peak, (n_in + int(x[0].numel())) * act_bits // 8)
                out[name] = {"params": int(params), "weight_B": int(params * weight_bits // 8),
                             "peak_act_B": int(peak)}
    finally:
        for h in handles:
            h.remove()
    last = stages[-1][0]
    out["classe"] = out[last]
    raw_B = S.n_channels * S.window * S.sample_bits // 8
    out["finestra"] = out["flusso"] = {"params": 0, "weight_B": 0, "peak_act_B": raw_B}
    return out


# ── collocare: vincoli, poi il costo ─────────────────────────────────────────

def place(rows, link: Link, node: dict | None = None, hops: int = 1, resources: dict | None = None,
          deadline_s: float | None = None, raw_allowed: bool = True,
          decision_period_s: float | None = None, server_macs_s: float = 2.4e9,
          link_cost_fn=None) -> list[dict]:
    """Every row with energy, delay and the reasons that exclude it.

    node is a dict like edge.compress.NODE (e_mac_J, macs_per_s, flash_B,
    ram_B; default NODE). Energy: mac_head e_MAC + h transmissions on link
    (link_cost, frames included). Delay: head on the node, h times the air
    time, tail on the server. A row is excluded when
      «memoria»   resources (head_resources) says the head does not fit flash or RAM;
      «scadenza»  the delay exceeds deadline_s;
      «grezzo»    raw_allowed is False and the row is the raw end;
      «duty»      the link has a duty cycle and its air time per decision
                  exceeds duty_cycle * decision_period_s (default: the step
                  of the active dataset over fs).
    Adds energy_J, t_s, frames, air_bits, excluded (list of reasons), feasible.
    The tests decide what is possible; choose() then takes the cheapest.

    link_cost_fn(link, bits, hops) -> dict like link_cost replaces the cost of
    the link: a project plugs in its own radio model (a BLE connection
    interval, MQTT headers, a measured energy per packet).
    """
    from iomtlab.edge.compress import NODE
    from iomtlab.spec import active
    n = {**NODE, **(node or {})}
    if decision_period_s is None:
        S = active()
        decision_period_s = S.step / S.fs_hz
    out = []
    for i, r in enumerate(rows):
        lc = (link_cost_fn or link_cost)(link, r["bits"], hops)
        e = None if lc["energy_J"] is None else r["mac_head"] * n["e_mac_J"] + lc["energy_J"]
        t = r["mac_head"] / n["macs_per_s"] + lc["t_air_s"] + r["mac_tail"] / server_macs_s
        why = []
        if resources is not None and r["name"] in resources:
            res = resources[r["name"]]
            if res["weight_B"] > n["flash_B"] or res["peak_act_B"] > n["ram_B"]:
                why.append("memoria")
        if deadline_s is not None and t > deadline_s:
            why.append("scadenza")
        if not raw_allowed and i == 0:
            why.append("grezzo")
        if link.duty_cycle is not None and lc["t_air_s"] / hops > link.duty_cycle * decision_period_s:
            why.append("duty")
        out.append({**r, "energy_J": e, "t_s": t, "frames": lc["frames"], "air_bits": lc["air_bits"],
                    "excluded": why, "feasible": not why})
    return out


def choose(placed) -> dict | None:
    """The feasible row of place() with the least energy (least delay if energy is unknown); None if none."""
    ok = [r for r in placed if r["feasible"]]
    if not ok:
        return None
    if any(r["energy_J"] is None for r in ok):
        return min(ok, key=lambda r: r["t_s"])
    return min(ok, key=lambda r: (r["energy_J"], r["t_s"]))
