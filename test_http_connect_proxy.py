from __future__ import annotations

import socket
import threading

from scripts.server.http_connect_proxy import _open_tunnel


def test_open_tunnel_sends_connect_and_preserves_remainder() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    request = bytearray()

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            while b"\r\n\r\n" not in request:
                request.extend(connection.recv(4096))
            connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        listener.close()

    worker = threading.Thread(target=serve)
    worker.start()
    tunnel = _open_tunnel(f"http://127.0.0.1:{port}", "ssh.example", 443)
    tunnel.close()
    worker.join(timeout=5)

    assert request.startswith(b"CONNECT ssh.example:443 HTTP/1.1\r\n")
