import math
from dataclasses import dataclass

from .syms import Symbol, FuncSymbol


float32 = FuncSymbol("float32", tuple(["val"]))

tanh = FuncSymbol("tanh", tuple(["val"]))
sqrt = FuncSymbol("sqrt", tuple(["val"]))

pi = math.pi
