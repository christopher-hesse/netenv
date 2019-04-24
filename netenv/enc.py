"""
Encoding/decoding functions
"""

import json
import base64

import gym.spaces
import numpy as np


def _dict_space_to_dict(dict_space):
    """
    Convert a dict space to a dictionary
    """
    assert isinstance(dict_space, gym.spaces.Dict)
    result = {"spaces": []}
    for name, space in dict_space.spaces.items():
        class_name = space.__class__.__name__
        r = dict(name=name, dtype=space.dtype.name, class_name=class_name)
        if class_name == "Box":
            r["low"] = space.low
            r["high"] = space.high
        elif class_name == "Discrete":
            r["n"] = space.n
        elif class_name == "MultiBinary":
            r["n"] = space.n
        else:
            assert False, "unrecognized space"
        result["spaces"].append(r)
    return result


def _dict_to_dict_space(dict_space_dict):
    """
    Convert a dict space dictionary back to a dict space
    """
    spaces = []
    for d in dict_space_dict["spaces"]:
        dtype = np.dtype(d["dtype"])
        class_name = d["class_name"]
        if class_name == "Box":
            space = gym.spaces.Box(
                low=np.asarray(d["low"]), high=np.asarray(d["high"]), dtype=dtype
            )
        elif class_name == "Discrete":
            space = gym.spaces.Discrete(n=d["n"])
            space.dtype = dtype
        elif class_name == "MultiBinary":
            space = gym.spaces.MultiBinary(n=d["n"])
            space.dtype = dtype
        else:
            assert False, "unrecognized space"
        spaces.append((d["name"], space))

    return gym.spaces.Dict(spaces)


class JSONEncoder(json.JSONEncoder):
    # https://github.com/PyCQA/pylint/issues/414
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, np.ndarray):
            return dict(
                __kind__="numpy",
                dtype=o.dtype.name,
                shape=o.shape,
                data=base64.b64encode(o.tobytes()).decode("utf8"),
            )
        elif isinstance(o, np.float32):
            return float(o)
        elif isinstance(o, np.uint8):
            return int(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, bytes):
            return dict(__kind__="bytes", data=base64.b64encode(o).decode("utf8"))
        else:
            return super().default(o)


def json_decoder(dct):
    if "__kind__" in dct:
        if dct["__kind__"] == "numpy":
            data = base64.b64decode(dct["data"].encode("utf8"))
            return np.frombuffer(buffer=data, dtype=np.dtype(dct["dtype"])).reshape(
                dct["shape"]
            )
        elif dct["__kind__"] == "bytes":
            return base64.b64decode(dct["data"].encode("utf8"))
        else:
            raise Exception("invalid kind")
    return dct


def encode_json(obj):
    """Encode an object as json, supports numpy arrays"""
    return json.dumps(obj, cls=JSONEncoder)


def decode_json(data):
    """Decode an object from json, supports numpy arrays"""
    return json.loads(data, object_hook=json_decoder)
