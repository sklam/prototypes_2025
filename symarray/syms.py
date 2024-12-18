from __future__ import annotations
from typing import Any

from dataclasses import dataclass
import inspect


def as_expr(val: Any) -> Expr:
    match val:
        case float(x):
            return FloatLiteral(x)
        case int(x):
            return IntLiteral(x)

        case Expr() as x:
            return x
        case _:
            raise TypeError(type(val))


@dataclass(frozen=True)
class Expr:
    def __add__(self, other: Expr) -> Expr:
        return FuncSymbol("Add", tuple(["lhs", "rhs"]))(self, other)

    def __mul__(self, other: Expr) -> Expr:
        return FuncSymbol("Mul", tuple(["lhs", "rhs"]))(self, other)

    def __truediv__(self, other: Expr) -> Expr:
        return FuncSymbol("Div", tuple(["lhs", "rhs"]))(self, other)

    def __pow__(self, other: Expr) -> Expr:
        return FuncSymbol("Pow", tuple(["lhs", "rhs"]))(self, other)


@dataclass(frozen=True)
class FloatLiteral(Expr):
    fval: float


@dataclass(frozen=True)
class IntLiteral(Expr):
    ival: float


@dataclass(frozen=True)
class Symbol(Expr):
    name: str


@dataclass(frozen=True)
class FuncSymbol(Symbol):
    params: tuple[str, ...]

    def __call__(self, *args, **kwargs) -> CallExpr:
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    k, kind=inspect._ParameterKind.POSITIONAL_ONLY
                )
                for k in self.params
            ]
        )
        ba = sig.bind(
            *map(as_expr, args), **{k: as_expr(v) for k, v in kwargs.items()}
        )
        return CallExpr(callee=self, args=tuple(ba.args))


@dataclass(frozen=True)
class CallExpr(Expr):
    callee: FuncSymbol
    args: tuple[Expr]
