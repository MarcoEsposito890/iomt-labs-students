"""A TCP forwarder that records what it relays, and a one-shot TCP receiver.

    client  ──►  Tap (localhost:port)  ──►  upstream (broker, or a Receiver)
            ◄──                        ◄──

The client connects to the tap instead of the server. The tap opens its own
connection upstream and copies bytes both ways unchanged, recording every
chunk with its connection number, direction and time.

Directions are named from the client's side: "up" is client to server,
"down" is server to client.

The recording is what the application gave to TCP, grouped as the tap's
recv() calls happened to return it. The grouping means nothing; the byte
counts are exact.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

__all__ = ["Tap", "Receiver", "broker_available", "mqtt_client", "Chunk"]


@dataclass(frozen=True)
class Chunk:
    """The bytes returned by one recv() of the tap."""
    conn: int
    direction: str   # "up" o "down"
    t: float         # time.perf_counter() all'arrivo
    data: bytes


def _listener(port: int = 0) -> socket.socket:
    """A listening TCP socket on 127.0.0.1; port 0 lets the OS pick one."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen()
    return s


class Tap:
    """Forward 127.0.0.1:<port> to upstream and record the traffic.

    >>> with Tap(("localhost", 1883)) as tap:
    ...     client.connect("127.0.0.1", tap.port)
    ...     ...
    ...     tap.packets("up")
    """

    def __init__(self, upstream=("localhost", 1883), port: int = 0):
        self.upstream = tuple(upstream)
        self._srv = _listener(port)
        self.port = self._srv.getsockname()[1]        # la porta che il sistema mi ha assegnato
        self.chunks: list[Chunk] = []
        self._lock = threading.Lock()
        self._n = 0
        self._socks: list[socket.socket] = []
        self._closed = False
        # accetto le connessioni in un thread a parte, così il notebook non si blocca
        threading.Thread(target=self._accept, daemon=True).start()

    # -- plumbing ----------------------------------------------------------
    def _accept(self):
        while not self._closed:
            try:
                c, _ = self._srv.accept()             # un client si collega al tap
            except OSError:
                return
            try:
                # per ogni client apro la mia connessione verso il server vero
                u = socket.create_connection(self.upstream, timeout=5)
                u.settimeout(None)
            except OSError:
                c.close()
                continue
            # disattivo Nagle: inoltro ogni pezzo appena arriva, senza accorparlo
            for s in (c, u):
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                k = self._n                           # numero la connessione
                self._n += 1
                self._socks += [c, u]
            # due thread per connessione: uno copia verso il server, l'altro verso il client
            threading.Thread(target=self._pipe, args=(c, u, k, "up"), daemon=True).start()
            threading.Thread(target=self._pipe, args=(u, c, k, "down"), daemon=True).start()

    def _pipe(self, a, b, k, d):
        try:
            while True:
                x = a.recv(65536)
                if not x:                             # il mittente ha chiuso
                    break
                with self._lock:
                    # registro il pezzo con connessione, direzione e istante di arrivo
                    self.chunks.append(Chunk(k, d, time.perf_counter(), x))
                b.sendall(x)                          # e lo inoltro identico dall'altra parte
        except OSError:
            pass
        finally:
            # propago la chiusura solo in scrittura: chi ha smesso di mandare può ancora
            # leggere. I due socket li chiudo in close(), non qui
            try:
                b.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def close(self):
        """Stop accepting and close every socket."""
        self._closed = True
        for s in [self._srv, *self._socks]:
            try:
                s.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- reading what was recorded -----------------------------------------
    @property
    def connections(self) -> list[int]:
        """Numbers of the connections seen so far, from 0."""
        return sorted({c.conn for c in self.chunks})

    def stream(self, direction: str, conn: int | None = None) -> bytes:
        """All bytes relayed in one direction, in order (one connection or all)."""
        with self._lock:
            # rimetto insieme i pezzi di quella direzione, nell'ordine in cui sono arrivati
            return b"".join(c.data for c in self.chunks
                            if c.direction == direction and (conn is None or c.conn == conn))

    def nbytes(self, direction: str, conn: int | None = None) -> int:
        """Number of bytes relayed in one direction."""
        return len(self.stream(direction, conn))

    def packets(self, direction: str, conn: int | None = None):
        """The MQTT control packets relayed in one direction (see split_packets).

        With conn=None each connection is split on its own and the results
        are concatenated, so no packet straddles two connections.
        """
        from iomtlab.net.mqtt_wire import split_packets
        conns = self.connections if conn is None else [conn]
        out = []
        for k in conns:
            # taglio in pacchetti il flusso di una connessione alla volta
            out += split_packets(self.stream(direction, k))
        return out


class Receiver:
    """Accept one TCP connection on 127.0.0.1 and run handler(sock) on it.

    The handler runs in a thread. result() waits for it and returns what it
    returned, or re-raises what it raised.
    """

    def __init__(self, handler, port: int = 0):
        self._srv = _listener(port)
        self.port = self._srv.getsockname()[1]
        self._out: dict = {}
        self._done = threading.Event()
        threading.Thread(target=self._run, args=(handler,), daemon=True).start()

    def _run(self, handler):
        try:
            c, _ = self._srv.accept()                 # aspetto un solo mittente
            with c:
                self._out["value"] = handler(c)       # e passo il socket alla funzione che legge
        except BaseException as e:            # noqa: BLE001
            # tengo l'eccezione per rilanciarla in result()
            self._out["error"] = e
        finally:
            self._srv.close()
            self._done.set()

    def result(self, timeout: float = 10.0):
        """Wait up to timeout seconds for the handler and return its value.

        Raises TimeoutError if the handler has not finished.
        """
        if not self._done.wait(timeout):
            raise TimeoutError("il ricevitore non ha finito: il mittente si è collegato "
                               "e ha chiuso il suo socket?")
        if "error" in self._out:
            raise self._out["error"]
        return self._out.get("value")


def broker_available(host: str = "localhost", port: int = 1883, timeout: float = 1.0) -> bool:
    """True if something accepts TCP connections at host:port within timeout seconds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def mqtt_client(client_id: str, port: int, host: str = "127.0.0.1",
                keepalive: int = 60, on_message=None, timeout: float = 5.0):
    """A connected paho-mqtt client speaking MQTT 3.1.1, with its network loop running.

    Wraps the paho 2.x setup. Raises ConnectionError if no CONNACK arrives
    within timeout seconds or the broker refuses. Call
    client.disconnect(); client.loop_stop() when done.
    """
    import paho.mqtt.client as mqtt

    # MQTT 3.1.1 con sessione pulita: il broker non tiene nulla fra una connessione e l'altra
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                    protocol=mqtt.MQTTv311, clean_session=True)
    ready = threading.Event()
    result = {}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        result["rc"] = reason_code                    # salvo l'esito del CONNACK
        ready.set()

    c.on_connect = on_connect
    if on_message is not None:
        c.on_message = on_message
    c.connect(host, port, keepalive=keepalive)        # mando il CONNECT
    # il ciclo di rete di paho gira in un thread suo, così le risposte arrivano mentre il notebook prosegue
    c.loop_start()
    if not ready.wait(timeout):                       # aspetto il CONNACK prima di restituire il client
        c.loop_stop()
        raise ConnectionError(f"nessun CONNACK da {host}:{port} entro {timeout} s")
    if result["rc"].is_failure:
        c.loop_stop()
        raise ConnectionError(f"il broker ha rifiutato la connessione: {result['rc']}")
    return c
