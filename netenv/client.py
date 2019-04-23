import mmap
import collections
import sys

import numpy as np
import gym.spaces

from . import net, enc, util


class Client:
    """
    Connect to a networked VecEnv provided by Server or a compatible implementation.

    Provides a VecEnv interface, only works with dictionary obs/act spaces
    """

    def __init__(
        self,
        addr,
        num_envs,
        socket_kind="tcp",
        env_options=None,
        reuse_arrays=False,
        use_shared_memory=False,
    ):
        self._addr = addr
        if env_options is None:
            env_options = {}
        self._env_options = env_options
        self._reuse_arrays = reuse_arrays
        self._use_shared_memory = use_shared_memory

        if self._use_shared_memory:
            assert (
                socket_kind == "unix"
                and sys.platform == "linux"
                and sys.maxsize > 2 ** 32
            ), "shared memory only enabled for 64-bit linux"

        sock = util._create_socket(socket_kind)
        sock.settimeout(util.SOCKET_TIMEOUT)
        sock.connect(self._addr)
        self._stream = net._Stream(sock)

        # server hello
        self._stream.sendall(net._HELLO)
        buf = bytearray(len(net._HELLO))
        self._stream.recvall_into(buf)
        assert (
            buf == net._HELLO
        ), f"invalid server hello, make sure that {addr} is the correct address"

        self.num_envs = num_envs

        env_data = {
            "num_envs": self.num_envs,
            "use_shared_memory": self._use_shared_memory,
            "env_options": env_options,
        }
        env_attrs = self._stream.request(net._CMD_INIT, env_data)

        self.metadata = env_attrs["metadata"]
        self.reward_range = env_attrs["reward_range"]
        self.spec = env_attrs["spec"]
        self._action_space = enc._dict_to_dict_space(env_attrs["_action_space"])
        self._step_space = enc._dict_to_dict_space(env_attrs["_step_space"])

        self.action_space, self._process_act = util._convert_dict_space(
            self._action_space, wrap=False, is_action=True
        )

        # pre-allocate buffers for step and action
        act_buf = None
        step_buf = None
        if self._use_shared_memory:
            # allocate buffers from shared memory
            act_space_size = util._calc_space_size(self.num_envs, self._action_space)
            step_space_size = util._calc_space_size(self.num_envs, self._step_space)
            fd = util._recv_fd(sock)
            buf = memoryview(
                mmap.mmap(
                    fileno=fd,
                    length=act_space_size + step_space_size,
                    flags=mmap.MAP_SHARED,  # pylint: disable=no-member
                )
            )
            act_buf = buf[:act_space_size]
            step_buf = buf[act_space_size:]

        self._allocated_act, self._allocated_act_buf = util._create_space_arrays(
            num_envs=self.num_envs, space=self._action_space, buf=act_buf
        )

        self._allocated_step, self._allocated_step_buf = util._create_space_arrays(
            num_envs=self.num_envs, space=self._step_space, buf=step_buf
        )

        self._rews = self._allocated_step["_rews"]
        self._dones = self._allocated_step["_dones"]

        # we use a single step space that includes obs, rews, dones (no info)
        # break out an observation space from that
        obs_spaces = list(self._step_space.spaces.items())[:-2]
        observation_space = gym.spaces.Dict(obs_spaces)
        self.observation_space, self._process_obs = util._convert_dict_space(
            observation_space, wrap=False, is_action=False
        )
        # slice out the observation dict from our step object
        self._obs = collections.OrderedDict()
        for name, _space in obs_spaces:
            self._obs[name] = self._allocated_step[name]

        # infos not supported
        self._infos = [{} for _ in range(self.num_envs)]

    def __repr__(self):
        return f"<Client addr={self._addr} env_options={self._env_options}>"

    def _maybe_copy_arrays(self, obj):
        """
        Copy arrays if we don't have reuse_arrays set
        """
        if self._reuse_arrays:
            return obj

        if isinstance(obj, np.ndarray):
            return obj.copy()

        r = collections.OrderedDict()
        for k, v in obj.items():
            r[k] = v.copy()
        return r

    def reset(self):
        self._stream.sendall(net._CMD_RESET)
        assert self._stream.recvall(1) == net._CMD_RESET
        if not self._use_shared_memory:
            self._stream.recvall_into(self._allocated_step_buf)

        obs = self._maybe_copy_arrays(self._obs)
        return self._process_obs(obs)

    def step_async(self, actions):
        actions = self._process_act(actions)
        for name in self._action_space.spaces:
            self._allocated_act[name][:] = actions[name]
        self._stream.sendall(net._CMD_STEP)
        if not self._use_shared_memory:
            self._stream.sendall(self._allocated_act_buf)

    def step_wait(self):
        assert self._stream.recvall(1) == net._CMD_STEP
        if not self._use_shared_memory:
            self._stream.recvall_into(self._allocated_step_buf)

        obs = self._maybe_copy_arrays(self._obs)
        obs = self._process_obs(obs)
        rews = self._maybe_copy_arrays(self._rews)
        dones = self._maybe_copy_arrays(self._dones)
        return obs, rews, dones, self._infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        resp = self._stream.request(net._CMD_RENDER, dict(mode=mode))
        return resp["result"]

    def close(self):
        self._stream.close()
