"""Moving a measurement off the node: raw TCP, a payload format, MQTT on the wire.

Used by Labs 0 and 1 and by the lab slides. Six modules:

  framing    frame, read_frame: length-prefixed messages on a TCP stream.
  payload    pack_window, unpack_window, encode_json, encode_float32,
             synthetic_window: one window of the course pipeline in the
             encodings compared in Lab 1.
  mqtt_wire  remaining_length, split_packets, publish_size: MQTT control
             packets read out of a byte stream.
  tap        Tap, Receiver, broker_available, mqtt_client: a TCP forwarder
             that records every byte it relays, a one-shot receiver, and the
             paho-mqtt setup.
  measure    the measurements of Lab 1, one function each.
  figures    the figures of Lab 1, drawn from what measure returns.

The tap sees the bytes the application hands to TCP. TCP, IP and link-layer
headers are not measured; where the lab needs them it computes them.
"""
from iomtlab.net.framing import HEADER, MAX_PAYLOAD, frame, read_frame  # noqa: F401
from iomtlab.net.payload import (                                       # noqa: F401
    WINDOW_HEADER, lsb, pack_window, unpack_window,
    encode_json, encode_float32, synthetic_window, DECISION, pack_decision, unpack_decision,
    )
from iomtlab.net.mqtt_wire import (                                     # noqa: F401
    PACKET_NAMES, Packet, encode_remaining_length, remaining_length,
    split_packets, publish_size,
)
from iomtlab.net.tap import Tap, Receiver, broker_available, mqtt_client  # noqa: F401
from iomtlab.net.measure import (                                       # noqa: F401
    BROKER, TOPIC, IPV6_TCP_HEADERS, settle, recv_sizes, framed_transfer,
    encoding_sizes, tcp_framed_cost, mqtt_cost, overhead_table, mqtt_path,
    path_table, fanout_table,
)
from iomtlab.net.figures import (                                       # noqa: F401
    plot_recv, plot_encodings, plot_overhead, plot_paths,
)

__all__ = [
    "HEADER", "MAX_PAYLOAD", "frame", "read_frame",
    "WINDOW_HEADER", "lsb", "pack_window", "unpack_window",
    "encode_json", "encode_float32", "synthetic_window",
    "PACKET_NAMES", "Packet", "encode_remaining_length", "remaining_length",
    "split_packets", "publish_size",
    "Tap", "Receiver", "broker_available", "mqtt_client",
    "BROKER", "TOPIC", "IPV6_TCP_HEADERS", "settle", "recv_sizes", "framed_transfer",
    "encoding_sizes", "tcp_framed_cost", "mqtt_cost", "overhead_table", "mqtt_path",
    "path_table", "fanout_table",
    "plot_recv", "plot_encodings", "plot_overhead", "plot_paths",
]
