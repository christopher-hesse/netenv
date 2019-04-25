from pkg_resources import get_distribution, DistributionNotFound

from .server import Server
from .client import Client

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


__all__ = ["Server", "Client"]
