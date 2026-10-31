"""The measurements of Lab 1, one function each.

Every function opens its own connections, sends, waits for the network
threads to deliver, and returns numbers. The notebook calls them with the
parameters the students change; the lab slides call them to draw the same
figures. What is measured is what the application hands to TCP (see Tap);
TCP, IP and link-layer headers are never measured, only computed where
stated.

    recv_sizes        raw TCP: what the receiver's recv() calls return
    framed_transfer   the same messages with a length prefix, through a Tap
    encoding_sizes    one window in five encodings, in bytes
    tcp_framed_cost   bytes per message of framed raw TCP
    mqtt_cost         bytes per message of MQTT, per packet type, at one QoS
    overhead_table    the two above for several payloads, plus the prediction
    mqtt_path         bytes per message on the node's link and towards the
                      subscribers
    path_table        mqtt_path for the three QoS levels
    fanout_table      mqtt_path for several numbers of subscribers
"""
from __future__ import annotations

import socket
import time

import numpy as np

from iomtlab.net.framing import frame, read_frame
from iomtlab.net.mqtt_wire import publish_size
from iomtlab.net.payload import encode_float32, encode_json, pack_window
from iomtlab.net.tap import Receiver, Tap, mqtt_client

__all__ = ["BROKER", "TOPIC", "IPV6_TCP_HEADERS", "settle", "recv_sizes",
           "framed_transfer", "encoding_sizes", "tcp_framed_cost", "mqtt_cost",
           "overhead_table", "mqtt_path", "path_table", "fanout_table"]

#: The local broker of the labs.
BROKER = ("localhost", 1883)
#: The topic of the theory slides, 18 characters.
TOPIC = "iomt/s01/sp/window"
#: Bytes of IPv6 (40) and TCP (20) headers per segment, without options.
#: Computed, not measured: the Tap cannot see them.
IPV6_TCP_HEADERS = 60
#: Packets that belong to one application message, up to QoS 2.
MSG_PACKETS = ("PUBLISH", "PUBACK", "PUBREC", "PUBREL", "PUBCOMP")


def settle(dt: float = 0.3) -> None:
    """Sleep dt seconds, so the network threads deliver what is in flight."""
    time.sleep(dt)


# ═══════════════════════════════════════════ raw TCP
def recv_sizes(n_msg: int = 10, size: int = 11, pause_s: float = 0.0,
               reader_wait_s: float = 0.3) -> list[int]:
    """Send n_msg messages of size bytes, one sendall() each, on one TCP connection.

    The sender pauses pause_s seconds between two messages. The receiver
    waits reader_wait_s seconds, then calls recv() until the sender closes.
    Returns the size in bytes of what each recv() returned. With pause_s = 0
    and reader_wait_s = 0 the result changes from run to run.
    """
    def reader(sock):
        settle(reader_wait_s)                          # il ricevitore aspetta prima di leggere
        sizes = []
        while (chunk := sock.recv(65536)):             # leggo finché il mittente non chiude
            sizes.append(len(chunk))                   # e mi segno quanti byte mi dà ogni recv
        return sizes

    rx = Receiver(reader)
    with socket.create_connection(("127.0.0.1", rx.port)) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # niente Nagle: ogni sendall parte subito
        for k in range(n_msg):
            s.sendall(bytes([65 + k % 26]) * size)    # messaggio k: size volte la stessa lettera
            if pause_s:
                time.sleep(pause_s)
    return rx.result()


def framed_transfer(msgs: list[bytes], split_writes: bool = False) -> dict:
    """Send msgs with frame() through a Tap to a receiver that uses read_frame().

    With split_writes the sender cuts every frame into three writes (1 byte,
    4 bytes, the rest) with a pause in between, so the receiver sees
    half-frames. Returns a dict with sent, received, identical,
    payload_bytes and wire_bytes (bytes on the connection).
    """
    def reader(sock):
        out = []
        while (m := read_frame(sock)) is not None:     # un frame alla volta, finché il mittente chiude
            out.append(m)
        return out

    rx = Receiver(reader)
    with Tap(("127.0.0.1", rx.port)) as tap:           # il tap sta in mezzo e conta i byte
        with socket.create_connection(("127.0.0.1", tap.port)) as s:
            for m in msgs:
                f = frame(m)                           # 2 byte di lunghezza, poi il messaggio
                if split_writes:
                    # spezzo apposta il frame in tre scritture: 1 byte, 4 byte, il resto
                    for part in (f[:1], f[1:5], f[5:]):
                        s.sendall(part)
                        settle(0.02)
                else:
                    s.sendall(f)
        got = rx.result()
        settle(0.1)
        wire = tap.nbytes("up")                        # byte passati dal mittente verso il ricevitore
    return {"sent": len(msgs), "received": len(got), "identical": got == list(msgs),
            "payload_bytes": sum(map(len, msgs)), "wire_bytes": wire}


# ═══════════════════════════════════════════ payload
def encoding_sizes(X: np.ndarray, t0_ms: int = 0, decimals: int = 3,
                   device: str = "sp") -> dict[str, int]:
    """Bytes of one window X of shape (n, 6) in five encodings, as a dict.

    json       compact JSON, every float with all its digits
    json_<d>   compact JSON rounded to d decimals
    float32    the samples as big-endian float32, no header
    int16      pack_window: 11-byte header, int16 against the full scale
    features   the 36 default features of iomtlab.features, float32
    """
    from iomtlab import features
    F, _ = features.extract(np.asarray(X)[None])      # X[None]: una finestra sola, asse delle finestre davanti
    return {
        "json": len(encode_json(0, t0_ms, X)),
        f"json_{decimals}": len(encode_json(0, t0_ms, X, decimals=decimals)),
        "float32": len(encode_float32(X)),
        "int16": len(pack_window(0, t0_ms, X, device)),
        "features": 4 * F.shape[1],                    # 4 byte per feature in float32
    }


# ═══════════════════════════════════════════ cost per message
def tcp_framed_cost(payload: bytes, n: int = 5) -> float:
    """Bytes per message on a raw TCP connection with frame(), averaged over n messages.

    The receiver never answers, so only the sender's direction counts.
    """
    r = framed_transfer([bytes(payload)] * n)
    if not r["identical"]:
        raise RuntimeError("il trasferimento con framing ha perso o cambiato un messaggio")
    return r["wire_bytes"] / n


def mqtt_cost(payload: bytes, qos: int = 0, topic: str = TOPIC, n: int = 5,
              broker=BROKER) -> dict[str, float]:
    """Bytes per message on the node's connection, per MQTT packet type, as a dict.

    One client connects through a Tap, publishes payload n times at qos and
    disconnects. CONNECT, CONNACK and DISCONNECT are left out: they are paid
    once per connection, not per message. For the int16 window at QoS 1 the
    result is {"PUBLISH": 636.0, "PUBACK": 4.0}.
    """
    with Tap(broker) as tap:
        c = mqtt_client(f"node-qos{qos}", tap.port)    # il client si collega al tap invece che al broker
        for _ in range(n):
            c.publish(topic, payload, qos=qos).wait_for_publish()
        settle()
        c.disconnect(); c.loop_stop(); settle()
        out: dict[str, float] = {}
        for d in ("up", "down"):                       # nodo verso broker, e ritorno
            for p in tap.packets(d):
                if p.name in MSG_PACKETS:
                    out[p.name] = out.get(p.name, 0) + p.size / n
    return out


def overhead_table(payloads: dict[str, bytes], topic: str = TOPIC, qos=(0, 1, 2),
                   n: int = 5):
    """Bytes per message on the node's connection, framed raw TCP against MQTT, as a DataFrame.

    One row per payload. Columns: payload_B; tcp (framed raw TCP); qos0,
    qos1, ... (MQTT, all per-message packets, both directions); publish_ok,
    True if every measured PUBLISH equals publish_size() from the
    specification.
    """
    import pandas as pd
    rows = []
    for name, p in payloads.items():
        r = {"payload": name, "payload_B": len(p), "tcp": tcp_framed_cost(p, n)}
        ok = True
        for q in qos:
            m = mqtt_cost(p, q, topic, n)
            r[f"qos{q}"] = sum(m.values())             # tutti i pacchetti del messaggio, nei due versi
            ok &= m.get("PUBLISH") == publish_size(len(topic), len(p), q)   # misurato contro calcolato
        r["publish_ok"] = ok
        rows.append(r)
    return pd.DataFrame(rows).set_index("payload")


# ═══════════════════════════════════════════ two hops, and fan-out
def mqtt_path(qos: int = 1, n_subs: int = 1, payload: bytes = b"0123456789",
              topic: str = TOPIC, n: int = 5, broker=BROKER) -> tuple[float, float]:
    """Bytes per message on the node's link and on the subscribers' side, as a pair.

    n_subs subscribers connect through one Tap and subscribe at qos; the
    publisher connects through another and publishes n times at qos.
    Returns (node link, all subscriber links together), per message,
    counting only the per-message packets in both directions.
    """
    def msg_bytes(tap):
        return sum(p.size for d in ("up", "down") for p in tap.packets(d)
                   if p.name in MSG_PACKETS)

    # due tap: uno sul collegamento del nodo, uno sul lato dei sottoscrittori
    with Tap(broker) as t_pub, Tap(broker) as t_sub:
        subs = [mqtt_client(f"sub{k}", t_sub.port) for k in range(n_subs)]
        for s in subs:
            s.subscribe(topic, qos=qos)
        settle()                                       # aspetto che le sottoscrizioni siano attive
        pub = mqtt_client("node", t_pub.port)
        for _ in range(n):
            pub.publish(topic, payload, qos=qos).wait_for_publish()
        settle(0.5)                                    # lascio al broker il tempo di consegnare a tutti
        for c in [pub, *subs]:
            c.disconnect(); c.loop_stop()
        settle()
        return msg_bytes(t_pub) / n, msg_bytes(t_sub) / n


def path_table(payload: bytes = b"0123456789", qos=(0, 1, 2), topic: str = TOPIC):
    """mqtt_path with one subscriber at each QoS, as a DataFrame: columns nodo, sottoscrittori, totale."""
    import pandas as pd
    rows = [dict(zip(("qos", "nodo", "sottoscrittori"), (q, *mqtt_path(q, 1, payload, topic))))
            for q in qos]
    df = pd.DataFrame(rows).set_index("qos")
    df["totale"] = df["nodo"] + df["sottoscrittori"]   # i due salti: nodo-broker e broker-sottoscrittore
    return df


def fanout_table(n_subs=(1, 2, 4), qos: int = 1, payload: bytes = b"0123456789",
                 topic: str = TOPIC):
    """mqtt_path for several numbers of subscribers, as a DataFrame: columns nodo, sottoscrittori."""
    import pandas as pd
    rows = [dict(zip(("sottoscrittori_n", "nodo", "sottoscrittori"),
                     (k, *mqtt_path(qos, k, payload, topic)))) for k in n_subs]
    return pd.DataFrame(rows).set_index("sottoscrittori_n")
