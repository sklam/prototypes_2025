import inspect

from functools import wraps

from .syms import Symbol


def lift(fn):
    sig = inspect.signature(fn)
    params = [k for k in sig.parameters]
    symbolic_params = [Symbol(name=k) for k in params]
    ba = sig.bind(*symbolic_params)
    exprtree = fn(*ba.args, **ba.kwargs)
    return exprtree
