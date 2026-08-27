"""Small typing compatibility helpers for the board's Python 3.7 runtime."""

try:
    from typing import Protocol
except ImportError:  # Python 3.7's stdlib typing has no Protocol.
    class Protocol(object):
        pass


__all__ = ["Protocol"]
