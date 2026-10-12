"""Scope observation statistics to valid examples, including empty DDP ranks."""
from contextlib import contextmanager
from contextvars import ContextVar

_VALID = ContextVar("valid_training_examples", default=None)


@contextmanager
def valid_moments(mask):
    token = _VALID.set(mask)
    try:
        yield
    finally:
        _VALID.reset(token)


def current_valid_mask():
    return _VALID.get()
