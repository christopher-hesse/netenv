import gym.spaces
import numpy as np

from . import enc
from .testing import make_fake_socket


def assert_spaces_equal(s1, s2):
    if isinstance(s1, gym.spaces.Dict):
        assert set(s1.spaces.keys()) == set(s2.spaces.keys())
        for name, ss1 in s1.spaces.items():
            ss2 = s2.spaces[name]
            assert_spaces_equal(ss1, ss2)
    else:
        assert s1.dtype == s2.dtype
        assert s1.shape == s2.shape
        if hasattr(s1, "low"):
            assert (s1.low == s2.low).all()
            assert (s1.high == s2.high).all()
        if hasattr(s1, "n"):
            assert s1.n == s2.n


def test_serialize():
    in_options = {"a": True, "b": 123}
    write_bio, write_s = make_fake_socket()
    write_s.send_dict(in_options)
    _read_bio, read_s = make_fake_socket(write_bio.getvalue())
    out_options = read_s.recv_dict()
    assert in_options == out_options

    space_tests = [
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
    for in_space in space_tests:
        write_bio, write_s = make_fake_socket()
        write_s.send_dict(enc._dict_space_to_dict(in_space))
        _read_bio, read_s = make_fake_socket(write_bio.getvalue())
        out_space = enc._dict_to_dict_space(read_s.recv_dict())
        assert_spaces_equal(in_space, out_space)


def test_json_coding():
    in_arr = np.array([1, 2, 3])
    out_arr = enc.decode_json(enc.encode_json(in_arr))
    assert (in_arr == out_arr).all()
    assert in_arr.shape == out_arr.shape
    assert in_arr.dtype == out_arr.dtype

    in_data = dict(
        a=float("-inf"),
        b=float("+inf"),
        c=np.float32("-inf"),
        d=np.arange(100, dtype=np.uint8).reshape((10, 10)),
        e=b"\x00\x01\x00",
    )
    out_data = enc.decode_json(enc.encode_json(in_data))
    for k in in_data.keys():
        assert np.array_equal(in_data[k], out_data[k])
