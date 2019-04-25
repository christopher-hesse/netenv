import socket
import struct

from . import enc

CMD_INIT = bytes([0])
CMD_RESET = bytes([1])
CMD_STEP = bytes([2])
CMD_RENDER = bytes([3])

_SOCKET_EXCEPTIONS = (ConnectionAbortedError, socket.timeout, BrokenPipeError)

HELLO = b"netenv_v1"

_uint32_struct = struct.Struct("!L")


class Stream:
    """
    Provide a few handy functions on top of a socket
    """

    def __init__(self, conn):
        self._conn = conn

    def sendall(self, buf):
        mv = memoryview(buf)
        try:
            self._conn.sendall(mv)
        except _SOCKET_EXCEPTIONS as e:
            raise EOFError() from e
        return mv.nbytes

    def recvall(self, length):
        """
        Read a byte string of the specified length or throw EOFError
        """
        buf = bytearray(length)
        mv = memoryview(buf)
        self.recvall_into(mv)
        return bytes(buf)

    def recvall_into(self, buf):
        """
        Read into the provided buffer until it is full or throw EOFError
        """
        mv = memoryview(buf)
        n = 0
        nbytes = mv.nbytes
        while n < nbytes:
            try:
                bytes_read = self._conn.recv_into(mv)
            except _SOCKET_EXCEPTIONS as e:
                raise EOFError() from e
            if bytes_read == 0:
                raise EOFError()
            n += bytes_read
            mv = mv[bytes_read:]
        return n

    def send_dict(self, d):
        """Send a dictionary by writing a JSON-encoded byte string prefixed with the length as a big-endian uint32 to stream"""
        msg = enc.encode_json(d).encode("utf8")
        self.sendall(_uint32_struct.pack(len(msg)))
        return self.sendall(msg)

    def recv_dict(self):
        """Receive a dictionary by reading a length-prefixed byte string"""
        len_buf = bytearray(4)
        self.recvall_into(len_buf)
        length = _uint32_struct.unpack(len_buf)[0]
        buf = bytearray(length)
        self.recvall_into(buf)
        msg = buf.decode("utf8")
        return enc.decode_json(msg)

    def request(self, cmd, req):
        """Perform a request to the server"""
        self.sendall(cmd)
        self.send_dict(req)
        resp_cmd = self.recvall(1)
        assert resp_cmd == cmd
        return self.recv_dict()

    def close(self):
        return self._conn.close()
