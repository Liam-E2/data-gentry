from inspect import signature
from typing import Callable, ParamSpec, TypeVar
from functools import wraps


A = ParamSpec("A")
R = TypeVar("R")


def with_requested_kwargs(fn: Callable[A, R], **kwargs) -> Callable[A, R]:
    @wraps(fn)
    def _inner(*args, **kwargs) -> R:
        return fn(*args, **{k: v for k, v in kwargs.items() if k in signature(fn).parameters})
    
    return _inner
