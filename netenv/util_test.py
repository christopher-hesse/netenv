import gym.spaces
import numpy as np

from . import util

ALIGNMENT = 64

TEST_SPACES = [
    gym.spaces.Dict(
        [
            ("s1", gym.spaces.Box(shape=(8, 8), low=0, high=128, dtype=np.uint8)),
            (
                "s2",
                gym.spaces.Box(
                    shape=(16, 16, 16), low=-1.0, high=1.0, dtype=np.float64
                ),
            ),
        ]
    ),
    gym.spaces.Dict([("s1", gym.spaces.MultiBinary(10))]),
    gym.spaces.Dict([("s1", gym.spaces.Discrete(12))]),
    gym.spaces.Dict(
        [("s1", gym.spaces.Box(shape=(8, 8), low=0, high=128, dtype=np.uint8))]
    ),
]


def test_alignment():
    spaces = [
        gym.spaces.Dict(
            [
                ("s1", gym.spaces.Box(shape=(8, 8), low=0, high=128, dtype=np.uint8)),
                (
                    "s2",
                    gym.spaces.Box(
                        shape=(16, 16, 16), low=-1.0, high=1.0, dtype=np.float64
                    ),
                ),
            ]
        ),
        gym.spaces.Dict([("s1", gym.spaces.MultiBinary(10))]),
        gym.spaces.Dict([("s1", gym.spaces.Discrete(12))]),
        gym.spaces.Dict(
            [("s1", gym.spaces.Box(shape=(8, 8), low=0, high=128, dtype=np.uint8))]
        ),
    ]
    for space in spaces:
        for buf in [bytearray(2 ** 20), None]:
            arrays, _buf = util.create_space_arrays(3, space, align=ALIGNMENT, buf=buf)
            for arr in arrays.values():
                assert arr.ctypes.data % ALIGNMENT == 0
