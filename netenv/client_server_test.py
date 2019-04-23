import threading
import pytest
import os
import functools
import sys

import numpy as np
import gym
import gym.spaces

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

from .client import Client
from .server import Server


def make_retro_env():
    import retro

    return retro.make("Airstriker-Genesis")


class NopEnvironment:
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            [("obs", gym.spaces.Box(low=0, high=3, shape=(2, 3), dtype=np.int32))]
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2, 3), dtype=np.float32
        )
        self.spec = None
        self.metadata = {}
        self._obs = self.observation_space.sample()

    def reset(self):
        return self._obs

    def step(self, action):
        return self._obs, 1.0, False, {}

    def close(self):
        pass


# from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/test_vec_env.py
class _SimpleEnv(gym.Env):
    """
    An environment with a pre-determined observation space
    and RNG seed.
    """

    def __init__(self, seed, shape, dtype, dict_obs_space):
        self._rand = np.random.RandomState(seed=seed)
        self._dict_obs_space = dict_obs_space
        self._dtype = dtype
        self._max_steps = seed + 1
        self._cur_obs = None
        self._cur_step = 0
        self.action_space = gym.spaces.Box(low=0, high=100, shape=shape, dtype=dtype)
        self._obs_keys = ["obs_a", "obs_b"]

        if self._dict_obs_space:
            spaces = []
            self._start_obs = {}
            for key in self._obs_keys:
                spaces.append((key, self.action_space))
                self._start_obs[key] = np.array(
                    self._rand.randint(0, 0x100, size=shape), dtype=dtype
                )
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            self.observation_space = self.action_space
            self._start_obs = np.array(
                self._rand.randint(0, 0x100, size=shape), dtype=dtype
            )

    def step(self, action):
        if self._dict_obs_space:
            for key in self._obs_keys:
                self._cur_obs[key] += np.array(action, dtype=self._dtype)
        else:
            self._cur_obs += np.array(action, dtype=self._dtype)
        self._cur_step += 1
        done = self._cur_step >= self._max_steps
        reward = self._cur_step / self._max_steps
        return self._cur_obs, reward, done, {}

    def reset(self):
        self._cur_obs = self._start_obs
        self._cur_step = 0
        return self._cur_obs

    def render(self, mode=None):
        assert mode == "rgb_array"
        return np.array(
            self._rand.randint(0, 0x100, size=(16, 8, 4)), dtype=self._dtype
        )


use_shared_memory_options = [True, False]
if sys.platform != "linux":
    use_shared_memory_options = [False]


@pytest.mark.parametrize("use_shared_memory", use_shared_memory_options)
@pytest.mark.parametrize("dict_obs_space", [True, False])
def test_simple_env(use_shared_memory, dict_obs_space):
    dtype = "uint8"
    shape = (3, 8)
    num_envs = 32
    num_steps = 1000
    make_envs = [
        functools.partial(_SimpleEnv, seed, shape, dtype, dict_obs_space=dict_obs_space)
        for seed in range(num_envs)
    ]
    np.random.seed(31337)

    env1 = DummyVecEnv(make_envs)

    if sys.platform == "win32":
        socket_kind = "tcp"
        addr = ("127.0.0.1", 0)
    else:
        socket_kind = "unix"
        addr = "/tmp/netenv.sock"
        if os.path.exists(addr):
            os.remove(addr)

    s = Server(
        addr=addr,
        socket_kind=socket_kind,
        make_venv=lambda num_envs: DummyVecEnv(make_envs),
    )
    addr = s.listen()
    t = threading.Thread(target=s.run, daemon=True)
    t.start()

    env2 = Client(
        addr=addr,
        socket_kind=socket_kind,
        num_envs=num_envs,
        env_options={},
        reuse_arrays=True,
        use_shared_memory=use_shared_memory,
    )
    actions = np.array(
        np.random.randint(0, 0x100, size=(num_envs,) + shape), dtype=dtype
    )

    def assert_arrays_close(arr1, arr2):
        assert np.array(arr1).shape == np.array(arr2).shape
        assert np.allclose(arr1, arr2)

    def assert_objs_close(obj1, obj2):
        if isinstance(obj1, dict):
            assert obj1.keys() == obj2.keys()
            for key in obj1.keys():
                assert_arrays_close(obj1[key], obj2[key])
        else:
            assert_arrays_close(obj1, obj2)

    try:
        obs1, obs2 = env1.reset(), env2.reset()
        assert_objs_close(obs1, obs2)
        for _ in range(num_steps):
            actions = np.array(
                np.random.randint(0, 0x100, size=(num_envs,) + shape), dtype=dtype
            )
            for env in [env1, env2]:
                env.step_async(actions)
            outs1 = env1.step_wait()
            outs2 = env2.step_wait()
            for out1, out2 in zip(outs1[:3], outs2[:3]):
                assert_objs_close(out1, out2)
            rend1, rend2 = env1.render(mode="rgb_array"), env2.render(mode="rgb_array")
            assert_objs_close(rend1, rend2)
    finally:
        env1.close()
        env2.close()


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.parametrize("use_shared_memory", use_shared_memory_options)
@pytest.mark.parametrize("make_env", [NopEnvironment, make_retro_env])
def test_unix_env_speed(use_shared_memory, make_env, benchmark):
    env_speed(
        "unix",
        make_env=make_env,
        benchmark=benchmark,
        use_shared_memory=use_shared_memory,
    )


@pytest.mark.parametrize("make_env", [NopEnvironment, make_retro_env])
def test_tcp_env_speed(make_env, benchmark):
    env_speed("tcp", make_env=make_env, benchmark=benchmark)


@pytest.mark.parametrize(
    "wrapper,make_env",
    [("DummyVecEnv", NopEnvironment), ("ShmemVecEnv", make_retro_env)],
)
def test_base_env_speed(wrapper, make_env, benchmark):
    env_speed(wrapper, make_env=make_env, benchmark=benchmark)


def env_speed(kind, make_env, benchmark, use_shared_memory=False):
    n = 2
    if kind == "DummyVecEnv":

        def make_venv():
            return DummyVecEnv([make_env] * n)

    elif kind == "ShmemVecEnv":

        def make_venv():
            return ShmemVecEnv([make_env] * n)

    else:
        socket_kind = kind
        # can't use dummyvecenv with retro since you can only have one instance per process
        vec_env_class = DummyVecEnv
        if make_env == make_retro_env:
            # can't use SubprocVecEnv because it doesn't handle dict observation spaces the same way as DummyVecEnv
            vec_env_class = ShmemVecEnv

        def make_server_venv(num_envs):
            return vec_env_class([make_env] * num_envs)

        if socket_kind == "unix":
            addr = "/tmp/netenv.sock"
            if os.path.exists(addr):
                os.remove(addr)
        elif socket_kind == "tcp":
            addr = ("127.0.0.1", 0)
        else:
            raise Exception("invalid socket_kind")

        s = Server(addr=addr, socket_kind=socket_kind, make_venv=make_server_venv)
        addr = s.listen()
        t = threading.Thread(target=s.run, daemon=True)
        t.start()

        def make_venv():
            return Client(
                addr=addr,
                socket_kind=socket_kind,
                num_envs=n,
                env_options={},
                reuse_arrays=True,
                use_shared_memory=use_shared_memory,
            )

    venv = make_venv()
    act = [venv.action_space.sample() for _ in range(n)]

    def rollout(max_steps):
        venv.reset()
        step_count = 0
        while step_count < max_steps:
            venv.step_async(act)
            obs, rews, dones, infos = venv.step_wait()
            step_count += 1

    benchmark(lambda: rollout(1000))
    venv.close()
