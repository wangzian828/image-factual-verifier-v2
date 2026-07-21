#!/usr/bin/env python3
"""Relay stdin/stdout through an HTTP CONNECT proxy for OpenSSH ProxyCommand."""

from __future__ import annotations

import argparse
import os
import select
import socket
import sys
from urllib.parse import urlsplit


def _open_tunnel(proxy: str, target_host: str, target_port: int) -> socket.socket:
    parsed = urlsplit(proxy)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("proxy must be an absolute http:// URL")
    proxy_port = parsed.port or 80
    connection = socket.create_connection((parsed.hostname, proxy_port), timeout=30)
    request = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        "Proxy-Connection: Keep-Alive\r\n\r\n"
    )
    connection.sendall(request.encode("ascii"))
    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = connection.recv(4096)
        if not chunk:
            raise ConnectionError("proxy closed before CONNECT response")
        response.extend(chunk)
        if len(response) > 65536:
            raise ConnectionError("proxy CONNECT response exceeded 64 KiB")
    header, remainder = bytes(response).split(b"\r\n\r\n", 1)
    status_line = header.splitlines()[0].decode("latin-1", errors="replace")
    parts = status_line.split(maxsplit=2)
    if len(parts) < 2 or parts[1] != "200":
        raise ConnectionError(f"proxy CONNECT failed: {status_line}")
    if remainder:
        os.write(sys.stdout.fileno(), remainder)
    connection.settimeout(None)
    return connection


def _relay(connection: socket.socket) -> None:
    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    socket_open = True
    stdin_open = True
    while socket_open:
        readers = [connection]
        if stdin_open:
            readers.append(stdin_fd)
        ready, _, _ = select.select(readers, [], [])
        if connection in ready:
            data = connection.recv(65536)
            if not data:
                socket_open = False
            else:
                os.write(stdout_fd, data)
        if stdin_open and stdin_fd in ready:
            data = os.read(stdin_fd, 65536)
            if not data:
                stdin_open = False
                connection.shutdown(socket.SHUT_WR)
            else:
                connection.sendall(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True)
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    connection = _open_tunnel(args.proxy, args.host, args.port)
    try:
        _relay(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
