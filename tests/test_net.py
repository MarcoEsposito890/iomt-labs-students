"""Lab 1: framing, window format, MQTT fixed header, and the tap as an instrument."""
import socket
import time

import numpy as np
import pytest

from iomtlab.net import (Receiver, Tap, broker_available, encode_json,
                         encode_remaining_length, frame, pack_window,
                         publish_size, read_frame, split_packets,
                         unpack_window)


def test_publish_size_matches_theory_deck():
    # B0_2_MQTT_theory, "One packet, byte by byte": 32 and 34 bytes
    assert publish_size(18, 10, 0) == 32
    assert publish_size(18, 10, 1) == 34
    # the Remaining Length grows to two bytes above 127
    assert publish_size(18, 611, 0) == 1 + 2 + 20 + 611


def test_split_packets_drops_an_incomplete_tail():
    pub = bytes([0x30]) + encode_remaining_length(4) + b"\x00\x01ab"
    ping = b"\xc0\x00"
    stream = pub + ping + pub[:3]                    # incomplete tail ignored
    ps = split_packets(stream)
    assert [p.name for p in ps] == ["PUBLISH", "PINGREQ"]
    assert [p.size for p in ps] == [6, 2]


def test_json_is_larger_than_int16():
    X = np.random.default_rng(0).normal(0, 3, (50, 6))
    assert len(encode_json(0, 0, X)) > len(encode_json(0, 0, X, decimals=3))


def test_tap_counts_raw_tcp_exactly():
    """The tap relays byte for byte and counts what it relays."""
    def sink(c):
        buf = bytearray()
        while (x := c.recv(4096)):
            buf += x
        return bytes(buf)

    rx = Receiver(sink)
    with Tap(("127.0.0.1", rx.port)) as tap:
        with socket.create_connection(("127.0.0.1", tap.port)) as s:
            for k in range(10):
                s.sendall(b"message %02d" % k)
        got = rx.result()
        time.sleep(0.1)
        assert got == b"".join(b"message %02d" % k for k in range(10))
        assert tap.nbytes("up") == 100 and tap.nbytes("down") == 0


def test_receiver_propagates_errors():
    def boom(c):
        raise RuntimeError("x")
    rx = Receiver(boom)
    socket.create_connection(("127.0.0.1", rx.port)).close()
    with pytest.raises(RuntimeError):
        rx.result()


@pytest.mark.skipif(not broker_available(), reason="nessun broker MQTT su localhost:1883")
def test_mqtt_through_tap_matches_spec():
    pytest.importorskip("paho.mqtt")
    from iomtlab.net import mqtt_client
    with Tap(("localhost", 1883)) as tap:
        c = mqtt_client("t-pub", tap.port)
        for qos in (0, 1):
            c.publish("iomt/s01/sp/window", b"x" * 10, qos=qos).wait_for_publish()
        time.sleep(0.2)
        c.disconnect(); c.loop_stop(); time.sleep(0.2)
        up = [p for p in tap.packets("up") if p.name == "PUBLISH"]
        assert [p.size for p in up] == [32, 34]
        assert [p.name for p in tap.packets("down")] == ["CONNACK", "PUBACK"]


# ── iomtlab.net.measure: the measurements of Lab 1 ─────────────────────────
def test_recv_sizes_merges_messages_when_the_reader_waits():
    from iomtlab.net import recv_sizes
    got = recv_sizes(n_msg=10, size=11, pause_s=0.0, reader_wait_s=0.3)
    assert sum(got) == 110
    assert len(got) < 10


def test_framed_transfer_survives_split_writes():
    from iomtlab.net import framed_transfer
    msgs = [b"x" * k for k in range(1, 8)]
    r = framed_transfer(msgs, split_writes=True)
    assert r["identical"] and r["received"] == len(msgs)
    assert r["wire_bytes"] == r["payload_bytes"] + 2 * len(msgs)


def test_encoding_sizes_of_the_course_window():
    from iomtlab.net import encoding_sizes, synthetic_window
    t0, X = synthetic_window()
    s = encoding_sizes(X, t0, decimals=3)
    assert s["int16"] == 11 + 2 * 50 * 6
    assert s["float32"] == 4 * 50 * 6
    assert s["features"] == 144
    assert s["json"] > s["json_3"] > s["float32"]


@pytest.mark.skipif(not broker_available(), reason="nessun broker MQTT su localhost:1883")
def test_overhead_and_paths_match_the_theory_deck():
    from iomtlab.net import fanout_table, overhead_table, path_table
    tab = overhead_table({"10 B": b"0123456789"}, n=2)
    assert tab["publish_ok"].all()
    assert tab.loc["10 B", ["tcp", "qos0", "qos1", "qos2"]].tolist() == [12, 32, 38, 46]
    # B0_4_MQTT_theory, «The three exchanges»: 64, 76, 92 bytes on both hops
    assert path_table()["totale"].tolist() == [64, 76, 92]
    fan = fanout_table((1, 3), qos=0)
    assert fan["nodo"].tolist() == [32, 32]
    assert fan["sottoscrittori"].tolist() == [32, 96]


def test_pack_decision_six_bytes_roundtrip():
    """The Lab 1 «Scrivi tu» reference: 6 bytes, big-endian, confidence in whole percent."""
    from iomtlab.net import pack_decision, unpack_decision
    b = pack_decision(123456, 2, 0.934)
    assert len(b) == 6 and b[:4] == (123456).to_bytes(4, "big")
    assert unpack_decision(b) == (123456, 2, 0.93)
    assert unpack_decision(pack_decision(2**32 + 5, 4, 1.7)) == (5, 4, 1.0)
