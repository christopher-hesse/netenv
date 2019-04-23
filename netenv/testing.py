import io

from . import net


class _FakeSocket:
    """
    Wrap a file with a socket-like interface

    Only has the minimal methods to work for testing purposes
    """

    def __init__(self, f):
        self._f = f

    def sendall(self, buf):
        return self._f.write(buf)

    def recv_into(self, buf):
        return self._f.readinto(buf)


def make_fake_socket(data=None):
    """
    Create a stream and associated BytesIO for testing purposes
    """
    bio = io.BytesIO(data)
    s = net._Stream(_FakeSocket(bio))
    return bio, s
