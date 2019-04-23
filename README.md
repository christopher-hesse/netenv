# netenv

OpenAI Gym Environment interface (VecEnv), but over a socket.  Features:

* Only supports numpy arrays for actions/observations (no infos)
* Efficient-ish, for Python
* Doesn't use pickle, so can work across languages, which is almost the only reason to make one of these
	
## Usage

### Server

```py
from netenv import Server

s = Server(
	addr=('127.0.0.1', 2345),
	make_venv=MyVecEnvClass,
)
s.run()
```

### Client

```py
from netenv import Client

env = Client(
	addr=('127.0.0.1', 2345),
	num_envs=16,
)
```

## Protocol

The client talks to the server using a request-response model, the requests are all of the form:

```
client: [command byte] [command-specific request payload]
server: [command byte] [command-specific response payload]
```

The server responds with the same command byte, but the response payload is different.

```
CMD_INIT(0)
Request: JSON-encoded dictionary of environment related data
Response: JSON-encoded dictionary of environment attributes including action space
	and a combined observation space and other spaces (called a step space)

CMD_RESET(1)
Request: <no payload>
Response: raw step data (obs, rews, dones)

CMD_STEP(2)
Request: raw action data
Response: raw step data

CMD_RENDER(3)
Request: JSON-encoded dictionary of kwargs to render function
Response: JSON-encoded dictionary of render response under the key `result`
```

Raw action and step data is the numpy array data as bytes.  Each individual numpy array is aligned on a 64 byte boundary.

See [`enc.py`](netenv/enc.py) for the format of the JSON-encoded numpy arrays as well as the JSON encoding for the space.


