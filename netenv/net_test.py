import socket
import threading
import tempfile
import os
import sys

import pytest

from . import net
from .testing import make_fake_socket


def read_msg(data, n):
    _bio, s = make_fake_socket(data)
    buf = bytearray(n)
    s.recvall_into(buf)
    return buf


def test_stream_errors():
    for mode in ["timeout", "close"]:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("localhost", 0))
        server.listen(1)
        addr = server.getsockname()

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(addr)
        server_sock, _addr = server.accept()
        server_stream = net.Stream(server_sock)
        if mode == "timeout":
            client_sock.settimeout(1)
        if mode == "close":
            server_stream.close()
        client_stream = net.Stream(client_sock)

        with pytest.raises(EOFError):
            if mode == "close":
                while True:
                    client_stream.sendall(b"hello")
            if mode == "timeout":
                client_stream.recvall(1)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_socket_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "test.sock")

        server = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM  # pylint: disable=no-member
        )
        server.bind(socket_path)
        server.listen(1)

        def server_loop():
            try:
                sock, _addr = server.accept()
                stream = net.Stream(sock)
                while True:
                    buf = bytearray(4096)
                    stream.recvall_into(buf)
                    stream.sendall(buf)
            except EOFError:
                pass

        t = threading.Thread(target=server_loop, daemon=True)
        t.start()

        client = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM  # pylint: disable=no-member
        )
        client.connect(socket_path)
        stream = net.Stream(client)
        data = bytearray(8192)
        for i in range(len(data)):
            data[i] = i % 256
        stream.sendall(data)

        buf = bytearray(8192)
        stream.recvall_into(buf)
        assert buf == data
