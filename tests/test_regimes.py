"""edge.regimes: the rule m + z < 1, the envelope, links, constraints."""
import math

import numpy as np
import pytest
import torch

from iomtlab.edge.cuts import cut_profile
from iomtlab.edge.regimes import (LINKS, Link, choose, envelope, head_resources, link_cost,
                                  lora_airtime_s, place, regime_map, rule)
from iomtlab.models import BottleneckCNN, ExampleCNN


def _rows(points):
    """Rows from (name, bits, mac_head) with the total MACs of the last."""
    M = points[-1][2]
    return [{"name": n, "values": b, "bits": b, "mac_head": m, "mac_tail": M - m} for n, b, m in points]


#: The example network of the theory, plus the bottleneck of the deck (c = 2 after pool1).
BOOK = _rows([("finestra", 4800, 0), ("collo", 400, 49600), ("pool3", 3072, 549760),
              ("dense", 512, 574336), ("classe", 3, 574656)])


def test_rule_matches_the_deck():
    r = {x["name"]: x for x in rule(BOOK)}
    assert r["pool3"]["m_plus_z"] > 1 and not r["pool3"]["can_win"]
    assert r["dense"]["m_plus_z"] > 1 and not r["dense"]["can_win"]
    c = r["collo"]
    assert c["m_plus_z"] < 1 and c["can_win"]
    assert c["k_lo"] == pytest.approx(49600 / 4400)
    assert c["k_hi"] == pytest.approx((574656 - 49600) / (400 - 3))


def test_rule_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 6)
        inner = [(f"c{i}", int(rng.integers(1, 9000)), int(rng.integers(1, 99999))) for i in range(n)]
        rows = _rows([("finestra", 4800, 0), *inner, ("classe", 3, 100000)])
        ks = np.geomspace(1e-3, 1e5, 4000)
        for x, r in zip(rule(rows)[1:-1], rows[1:-1]):
            beats = [r["mac_head"] + k * r["bits"] < min(k * 4800, 100000 + 3 * k) for k in ks]
            if any(beats):
                assert x["can_win"]


def test_envelope_is_the_minimum_everywhere():
    segs = envelope(BOOK)
    assert [s["name"] for s in segs] == ["finestra", "collo", "classe"]
    for k in np.geomspace(0.01, 5e5, 500):
        best = min(BOOK, key=lambda r: r["mac_head"] + k * r["bits"])["name"]
        seg = next(s for s in segs if s["k_from"] <= k <= s["k_to"])
        assert seg["name"] == best


def test_excluding_the_class_lets_an_interior_cut_win():
    segs = envelope([r for r in BOOK if r["name"] != "collo"], exclude=("classe",))
    assert segs[-1]["name"] == "dense"
    assert segs[-1]["k_from"] == pytest.approx(574336 / (4800 - 512))
    m = regime_map(BOOK, [400], exclusions=[(), ("collo", "classe", "dense")])
    assert [x["winner"] for x in m] == ["collo", "pool3"]


def test_hops_and_ratio_are_one_axis():
    # k = rho h: 802.15.4 (400 MAC/bit) at 3 hops is beyond the bottleneck interval, at 1 hop inside
    c = next(x for x in rule(BOOK) if x["name"] == "collo")
    assert c["k_lo"] < 400 < c["k_hi"] < 4 * 400


def test_lora_airtime_against_the_deck():
    assert lora_airtime_s(15, 12) == pytest.approx(1.155072)
    assert lora_airtime_s(15, 7) == pytest.approx(0.046336)


def test_link_cost_frames_and_energy():
    L = LINKS["802.15.4"]
    assert L.e_bit_J == pytest.approx(240e-9)
    c = link_cost(L, 6144)                       # 768 B: 7 frames of 116 B
    assert c["frames"] == 7 and c["air_bits"] == 8 * (768 + 7 * 17)
    assert link_cost(L, 6144, hops=3)["energy_J"] == pytest.approx(3 * c["energy_J"])
    assert link_cost(LINKS["lora_sf12"], 8)["energy_J"] is None


def test_stream_halves_the_raw_end_with_half_overlap():
    torch.manual_seed(0)
    m = ExampleCNN()
    w, s = cut_profile(m), cut_profile(m, raw="stream")
    assert s[0]["name"] == "flusso" and w[0]["name"] == "finestra"
    assert s[0]["bits"] == w[0]["bits"] // 2
    assert [r["bits"] for r in w[1:]] == [r["bits"] for r in s[1:]]
    with pytest.raises(ValueError):
        cut_profile(m, raw="boh")


def test_head_resources_grow_along_the_network():
    torch.manual_seed(0)
    m = ExampleCNN()
    res = head_resources(m)
    names = [n for n, _ in m.stages()]
    params = [res[n]["params"] for n in names]
    assert params == sorted(params) and params[-1] == sum(p.numel() for p in m.parameters())
    assert res["classe"]["params"] == params[-1] and res["finestra"]["params"] == 0


def test_place_excludes_then_choose_takes_the_cheapest():
    torch.manual_seed(0)
    m = ExampleCNN()
    rows, res = cut_profile(m), head_resources(m)
    # a fast radio: all on the server even if the model fits (the decision sequence would pick the node)
    assert choose(place(rows, LINKS["qpart"], resources=res))["name"] == "finestra"
    # a node too small for the whole model, raw data not allowed out: an interior cut
    small = {"flash_B": 40_000}
    c = choose(place(rows, LINKS["802.15.4"], node=small, resources=res, raw_allowed=False))
    assert c["name"] not in ("finestra", "classe")
    # LoRa SF12 every 0.5 s: nothing fits the duty cycle
    assert choose(place(rows, LINKS["lora_sf12"])) is None


def test_bottleneck_network_has_a_winning_cut():
    torch.manual_seed(0)
    r = {x["name"]: x for x in rule(cut_profile(BottleneckCNN()))}
    assert r["bottleneck"]["can_win"] and r["bottleneck"]["k_lo"] < 400 < r["bottleneck"]["k_hi"]


def test_ble_link_from_denni():
    L = LINKS["ble_denni"]
    assert L.e_bit_J == pytest.approx(1 / (350 * 1024 * 8))
    assert L.rate_bps == pytest.approx(8 / 11.6e-6)


def test_score_fn_hook_reproduces_the_default():
    from iomtlab.uncertainty.conformal import conformal_sets
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), 200)
    y = rng.integers(0, 5, 200)
    default = conformal_sets(p[:100], y[:100], p[100:], 0.1)
    same = conformal_sets(p[:100], y[:100], p[100:], 0.1,
                          score_fn=lambda q, yy: 1 - q[np.arange(len(yy)), yy])
    assert np.array_equal(default, same)


def test_link_cost_fn_hook_is_used():
    rows = BOOK
    flat = lambda link, bits, hops: {"frames": 1, "air_bits": bits, "t_air_s": 0.0, "energy_J": 1.0}
    out = place(rows, LINKS["802.15.4"], link_cost_fn=flat)
    assert all(r["energy_J"] >= 1.0 for r in out)


def test_fusion_rules_and_costs():
    from iomtlab.edge.fusion import (consensus_cost, fuse, fuse_confidence, fuse_mean,
                                     gateway_cost, score_bits, softmax)
    from iomtlab.fl.consensus import ring_graph
    a = np.array([[0.9, 0.1], [0.5, 0.5]]); b = np.array([[0.6, 0.4], [0.1, 0.9]])
    assert np.allclose(fuse([a, b]), [[0.75, 0.25], [0.3, 0.7]])
    c = fuse([a, b], fuse_confidence)
    assert np.allclose(c.sum(1), 1) and c[0, 0] > 0.75
    with pytest.raises(ValueError):
        fuse([a, b], lambda ps: ps[0] * 2)
    assert np.allclose(softmax(np.zeros((3, 4))).sum(1), 1)
    assert score_bits(5) == 40
    L = LINKS["802.15.4"]
    k = consensus_cost(ring_graph(8), 10, 40, L)
    assert k["frames_node"] == 30 and k["energy_J_total"] == pytest.approx(8 * k["energy_J_node"])
    assert consensus_cost(ring_graph(8), 10, 40, L, rx=False)["frames_node"] == 10
    g = gateway_cost(8, 40, L, hops=3)
    assert g["energy_J_node"] == pytest.approx(6 * link_cost(L, 40)["energy_J"])


def test_paired_windows_line_up(tmp_path, monkeypatch):
    from iomtlab.signal import build_paired_windows
    try:
        d = build_paired_windows()
    except (ValueError, FileNotFoundError):
        pytest.skip("dati assenti")
    assert d["X_sp"].shape == d["X_sw"].shape and len(d["y"]) == len(d["X_sp"])


def test_align_by_group():
    from iomtlab.edge.fusion import align_by_group
    a = np.array([[1], [2], [3]]); b = np.array([[30], [10]])
    out, g = align_by_group({"a": a, "b": b}, {"a": ["e1", "e2", "e3"], "b": ["e3", "e1"]})
    assert list(g) == ["e1", "e3"] and out[0].ravel().tolist() == [1, 3] and out[1].ravel().tolist() == [10, 30]


def test_bottleneck_position_moves_m():
    torch.manual_seed(0)
    from iomtlab.models.contract import check_model
    a, b = BottleneckCNN(at="pool1", bottleneck=2), BottleneckCNN(at="pool2", bottleneck=4)
    ra = {x["name"]: x for x in rule(cut_profile(a))}["bottleneck"]
    rb = {x["name"]: x for x in rule(cut_profile(b))}["bottleneck"]
    assert cut_profile(a)[3]["name"] == "bottleneck" and ra["m"] < 0.1 < rb["m"]
    assert ra["m_plus_z"] == pytest.approx(49600 / 577856 + 400 / 4800, rel=1e-6)
    x = torch.randn(3, 6, 50)
    for m in (a, b):
        from iomtlab.edge.split import split_model
        h, tl = split_model(m.eval(), "bottleneck")
        with torch.no_grad():
            assert torch.allclose(tl(h(x)), m(x))
    with pytest.raises(ValueError):
        BottleneckCNN(at="conv3")


def test_bottleneck_code_option():
    m = BottleneckCNN(bottleneck=2, at="pool1", code="linear")
    assert isinstance(m.relu_b, torch.nn.Identity) and m.config()["code"] == "linear"
    with pytest.raises(ValueError):
        BottleneckCNN(code="tanh")


def test_code_entropy():
    from iomtlab.edge.cuts import code_entropy
    torch.manual_seed(0)
    m = BottleneckCNN(bottleneck=2, at="pool1", code="linear")
    r = code_entropy(m, "bottleneck", torch.randn(20, 6, 50))
    assert r["values"] == 50 and r["bits_fixed"] == 400
    assert 0 < r["entropy_per_value"] <= 8 and r["bits_entropy"] <= 400
