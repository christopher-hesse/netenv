import threading
import mmap
import os
import shutil

import gym.spaces
import numpy as np

from . import enc, net, util

_REW_SPACE = gym.spaces.Box(
    low=float("-inf"), high=float("+inf"), shape=(), dtype=np.float32
)
_DONE_SPACE = gym.spaces.Box(low=False, high=True, shape=(), dtype=np.bool)


class _ConnectionHandler:
    """
    Handle a single connection for the server

    Creates and manages a single VecEnv corresponding to the connection
    """

    def __init__(self, conn, make_venv):
        self._conn = conn
        self._make_venv = make_venv
        self._venv = None
        self._tmpdir = None
        self._stream = None
        self._use_shared_memory = None
        self._act = None
        self._step = None
        self._act_buf = None
        self._step_buf = None
        self._obs_spaces = None

    def init(self):
        assert self._stream.recvall(1) == net.CMD_INIT
        self._stream.sendall(net.CMD_INIT)
        env_data = self._stream.recv_dict()
        self._use_shared_memory = env_data["use_shared_memory"]
        num_envs = env_data["num_envs"]

        self._venv = self._make_venv(num_envs=num_envs, **env_data["env_options"])
        if not isinstance(
            self._venv.observation_space, gym.spaces.Dict
        ) or not isinstance(self._venv.action_space, gym.spaces.Dict):
            self._venv = util.DictWrapper(self._venv)

        # the self._step space contains obs, rews, dones spaces
        self._obs_spaces = list(self._venv.observation_space.spaces.items())
        step_spaces = self._obs_spaces.copy()
        step_spaces.append(("_rews", _REW_SPACE))
        step_spaces.append(("_dones", _DONE_SPACE))
        step_space = gym.spaces.Dict(step_spaces)

        env_attrs = {
            "spec": None,  # not supported
            "reward_range": getattr(
                self._venv, "reward_range", (-float("inf"), float("inf"))
            ),
            "metadata": getattr(self._venv, "metadata", {"render.modes": []}),
            "_action_space": enc.dict_space_to_dict(self._venv.action_space),
            "_step_space": enc.dict_space_to_dict(step_space),
        }
        self._stream.send_dict(env_attrs)

        act_buf = None
        step_buf = None
        if self._use_shared_memory:
            act_space_size = util.calc_space_size(num_envs, self._venv.action_space)
            step_space_size = util.calc_space_size(num_envs, step_space)
            total_size = act_space_size + step_space_size
            # there are at least a few options for shared memory on linux:
            # file-backed mmap:
            #   you can easily create a file-backed mmap and have the other process open it
            #   but it might be nice to avoid having the file, which technically the OS
            #   should flush periodically
            # anonymous mmap:
            #   you can easily create an anonymous mmap, but it has no file descriptor to send to
            #   the other process.  if the server is a child of the client that's fine since it
            #   can inherit the mapping.  you can also map /dev/zero to create a fd, but sharing
            #   that fd doesn't seem to work even with MAP_SHARED
            # memfd_create mmap:
            #   memfd_create will create a fd but it's backed by ram instead of a file
            #   unfortunately it seems to require some ctypes stuff to make the syscall
            # posix ipc:
            #   this is probably the most reasonable way, but requires the posix_ipc package
            fd = util.memfd_create("netenv-shared-memory")
            assert os.write(fd, b"\x00" * total_size) == total_size
            mm = mmap.mmap(
                fileno=fd,
                length=total_size,
                flags=mmap.MAP_SHARED,  # pylint: disable=no-member
            )
            buf = memoryview(mm)
            util.send_fd(self._conn, fd)
            act_buf = buf[:act_space_size]
            step_buf = buf[act_space_size:]

        self._act, self._act_buf = util.create_space_arrays(
            num_envs, self._venv.action_space, buf=act_buf
        )
        self._step, self._step_buf = util.create_space_arrays(
            num_envs, step_space, buf=step_buf
        )

    def handle(self):
        try:
            self._handle()
        except EOFError:
            pass
        finally:
            if self._venv is not None:
                self._venv.close()
            self._conn.close()
            if self._tmpdir is not None:
                shutil.rmtree(self._tmpdir)

    def _handle(self):
        self._conn.settimeout(util.SOCKET_TIMEOUT)
        self._stream = net.Stream(self._conn)

        # exchange hellos with the client to make sure we are connected correctly
        buf = bytearray(len(net.HELLO))
        self._stream.recvall_into(buf)
        assert (
            buf == net.HELLO
        ), f"invalid client hello, make sure that client is correct version"
        self._stream.sendall(net.HELLO)

        self.init()

        while True:
            cmd = self._stream.recvall(1)

            if cmd == net.CMD_RESET:
                # copy the obs into our step object and send the buffer
                # for the step object
                obs = self._venv.reset()
                for name, _space in self._obs_spaces:
                    self._step[name][:] = obs[name]

                self._stream.sendall(net.CMD_RESET)
                if not self._use_shared_memory:
                    self._stream.sendall(self._step_buf)

            elif cmd == net.CMD_STEP:
                if not self._use_shared_memory:
                    self._stream.recvall_into(self._act_buf)

                self._venv.step_async(self._act)
                obs, rews, dones, _infos = self._venv.step_wait()

                # copy the results of the step into our step object
                for name, _space in self._obs_spaces:
                    self._step[name][:] = obs[name]
                self._step["_rews"][:] = rews
                self._step["_dones"][:] = dones

                self._stream.sendall(net.CMD_STEP)
                if not self._use_shared_memory:
                    self._stream.sendall(self._step_buf)

            elif cmd == net.CMD_RENDER:
                render_args = self._stream.recv_dict()
                result = self._venv.render(mode=render_args["mode"])
                self._stream.sendall(net.CMD_RENDER)
                render_args = self._stream.send_dict(dict(result=result))

            else:
                raise Exception("invalid command from client")


class Server:
    """
    Serve instances of the specified VecEnv with dictionary obs/self._act spaces on the given socket.

    Each connection will create its own VecEnv and run in its own thread.
    """

    def __init__(self, addr, make_venv, socket_kind="tcp"):
        self._addr = addr
        self._socket_kind = socket_kind
        self._make_venv = make_venv
        self._sock = None

    def __repr__(self):
        return f"<Server addr={self._addr}>"

    def listen(self):
        """
        Start listening for connections before run() is executed
        This is useful when binding to a dynamic port.
        """
        self._sock = util.create_socket(self._socket_kind)
        self._sock.bind(self._addr)
        return self._sock.getsockname()

    def run(self):
        """
        Start the server and wait forever for connections
        """
        if self._sock is None:
            self.listen()

        with self._sock:
            self._sock.listen(1)
            while True:
                conn, _addr = self._sock.accept()
                handler = _ConnectionHandler(conn=conn, make_venv=self._make_venv)
                t = threading.Thread(target=handler.handle)
                t.start()
