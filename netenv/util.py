import array
import collections
import ctypes
import platform
import socket
import sys

import gym.spaces
import numpy as np

SOCKET_TIMEOUT = 10
ALIGNMENT = 64
SEND_FD_MSG = b"send_fd"


def _align_memoryview(mv, align=ALIGNMENT):
    """
    Align a memoryview to a certain byte alignment but truncating the start and end of the memoryview,
    so that the resulting memoryview is always the same size regardless of alignment
    """
    address = ctypes.addressof(ctypes.c_byte.from_buffer(mv))
    start_amount = address % align
    start = 0 if start_amount == 0 else (align - start_amount)
    end_amount = (address + mv.nbytes) % align
    end = None if end_amount == 0 else -end_amount
    return mv[start:end]


def _calc_space_size(num_envs, space, align=ALIGNMENT):
    """
    Calculate the size of a space for the given number of envs, including padding for memory alignment
    """
    assert isinstance(space, gym.spaces.Dict)
    space_size = 0
    for subspace in space.spaces.values():
        shape = (num_envs,) + subspace.shape
        subspace_size = np.prod(shape) * subspace.dtype.itemsize
        # if the subspace doesn't end exactly on an align boundary, add padding
        subspace_align = subspace_size % align
        padding = 0 if subspace_align == 0 else (align - subspace_align)
        space_size += subspace_size + padding
    # the initial allocation can be off by [0, align) bytes, so make the space larger to compensate
    return space_size + (align - 1)


def _create_space_arrays(num_envs, space, buf=None, align=ALIGNMENT):
    """
    Create arrays for a space, returns an OrderedDict of numpy arrays along with a backing bytearray buffer

    The buffer may be provided and will be used instead of allocating a new buffer
    """
    if buf is None:
        buf = bytearray(_calc_space_size(num_envs, space, align=align))
    mv = memoryview(buf)
    aligned_mv = _align_memoryview(mv, align=align)
    offset = 0
    result = collections.OrderedDict()
    for name, subspace in space.spaces.items():
        shape = (num_envs,) + subspace.shape
        count = np.prod(shape)
        size = count * subspace.dtype.itemsize
        assert len(buf) >= size
        result[name] = np.frombuffer(
            buffer=aligned_mv, dtype=subspace.dtype, count=count, offset=offset
        ).reshape(shape)
        # pad the offset to a multiple of align
        space_align = size % align
        padding = 0 if space_align == 0 else (align - space_align)
        offset += size + padding
        assert offset % align == 0

    return result, aligned_mv


def _create_socket(kind):
    """
    Create a socket of the specified kind
    """
    if kind == "tcp":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if platform.system() == "Linux":
            # on linux, there will be a long delay without this option
            # it is 200x slower than mac for the benchmark
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    elif kind == "unix":
        s = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM  # pylint: disable=no-member
        )
    else:
        raise Exception(f"invalid socket kind {kind}")
    return s


def _recv_fd(sock):
    """
    Receive a file descriptor over a unix domain socket

    Based on https://docs.python.org/3/library/socket.html#socket.socket.recvmsg
    """
    fds = array.array("i")
    msg, ancdata, _flags, _addr = sock.recvmsg(
        len(SEND_FD_MSG), socket.CMSG_LEN(fds.itemsize)  # pylint: disable=no-member
    )
    assert msg == SEND_FD_MSG
    assert len(ancdata) == 1
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        assert (
            cmsg_level == socket.SOL_SOCKET
            and cmsg_type == socket.SCM_RIGHTS  # pylint: disable=no-member
        )
        # Append data, ignoring any truncated integers at the end.
        fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
    return fds[0]


def _send_fd(sock, fd):
    """
    Send a file descriptor over a unix domain socket

    Based on https://docs.python.org/3/library/socket.html#socket.socket.sendmsg
    """
    return sock.sendmsg(
        [SEND_FD_MSG],
        [
            (
                socket.SOL_SOCKET,
                socket.SCM_RIGHTS,  # pylint: disable=no-member
                array.array("i", [fd]),
            )
        ],
    )


def _memfd_create(name):
    """
    Create an anonymous memory backed file descriptor on Linux

    http://man7.org/linux/man-pages/man2/memfd_create.2.html
    """
    # make sure we are on 64 bit linux, otherwise this is probably the wrong syscall id
    assert sys.platform == "linux" and sys.maxsize > 2 ** 32
    SYS_memfd_create = 319
    libc = ctypes.CDLL(None)
    flags = 0
    return libc.syscall(SYS_memfd_create, name, flags)


def _space_is_wrapped(space):
    if isinstance(space, gym.spaces.Dict):
        spaces = list(space.spaces.items())
        return len(spaces) == 1 and spaces[0][0] == "_"
    else:
        return False


def _convert_dict_space(in_space, wrap, is_action):
    """
    Convert/unconvert a simple space to a dictionary space

    Args:
        in_space: input space to convert to a dict space
        wrap: True if we are wrapping, False for unwrapping
        is_action: True if this is an action (input), False for observation (output)

    Returns:
        out_space: output space after conversion
        process_fn: function to convert instances of the space to the correct form
    """

    def identity(sample):
        return sample

    def dictify(sample):
        return dict(_=sample)

    def undictify(sample):
        return sample["_"]

    if wrap:
        if isinstance(in_space, gym.spaces.Dict):
            out_space = in_space
            process_fn = identity
        else:
            out_space = gym.spaces.Dict([("_", in_space)])
            process_fn = undictify if is_action else dictify
    else:
        assert isinstance(in_space, gym.spaces.Dict)
        spaces = list(in_space.spaces.items())
        if _space_is_wrapped(in_space):
            out_space = spaces[0][1]
            process_fn = dictify if is_action else undictify
        else:
            out_space = in_space
            process_fn = identity
    return out_space, process_fn


class _DictWrapper:
    """
    Wrap or unwrap simple observation/action spaces on a VecEnv as dict spaces
    """

    def __init__(self, env, mode="wrap"):
        self._env = env

        self.action_space, self._process_act = _convert_dict_space(
            self._env.action_space, wrap=True, is_action=True
        )
        self.observation_space, self._process_obs = _convert_dict_space(
            self._env.observation_space, wrap=True, is_action=False
        )

    def reset(self):
        return self._process_obs(self._env.reset())

    def step_async(self, act):
        return self._env.step_async(self._process_act(act))

    def step_wait(self):
        obs, rews, dones, infos = self._env.step_wait()
        return self._process_obs(obs), rews, dones, infos

    def step(self, act):
        self.step_async(act)
        return self.step_wait()

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()
