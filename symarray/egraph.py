from __future__ import annotations

import egglog

from .syms import Expr, CallExpr, FloatLiteral, Symbol, IntLiteral

from egglog import f64Like, f64, StringLike, String, i64, i64Like, method


class Term(egglog.Expr):
    @classmethod
    def lit_f64(self, v: f64Like) -> Term: ...
    @classmethod
    def lit_f32(self, v: f64Like | i64) -> Term: ...
    @classmethod
    def lit_i64(self, v: i64Like) -> Term: ...
    @classmethod
    def var(self, k: StringLike) -> Term: ...

    # @method(unextractable=True)
    def __add__(self, other: Term) -> Term: ...

    # @method(unextractable=True)
    def __mul__(self, other: Term) -> Term: ...

    # @method(unextractable=True)
    def __truediv__(self, other: Term) -> Term: ...

    # @method(unextractable=True)
    def __pow__(self, other: Term) -> Term: ...


@egglog.function(cost=100)
def Sqrt(x: Term) -> Term: ...


@egglog.function(cost=10000)
def Tanh(x: Term) -> Term: ...
@egglog.function(cost=1000)
def Pow(x: Term, y: Term) -> Term: ...
@egglog.function(cost=1000)
def PowConst(x: Term, i: i64Like) -> Term: ...


@egglog.function
def Mul(x: Term, y: Term) -> Term: ...
@egglog.function(cost=2)
def Div(x: Term, y: Term) -> Term: ...
@egglog.function
def Add(x: Term, y: Term) -> Term: ...
@egglog.function
def CastF32(x: Term) -> Term: ...


def as_egraph(expr: Expr):
    match expr:
        case CallExpr() as expr:
            match expr.callee.name:
                case "sqrt":
                    return Sqrt(*map(as_egraph, expr.args))
                case "tanh":
                    return Tanh(*map(as_egraph, expr.args))
                case "Pow":
                    return Pow(*map(as_egraph, expr.args))
                case "Mul":
                    return Mul(*map(as_egraph, expr.args))
                case "Div":
                    return Div(*map(as_egraph, expr.args))
                case "Add":
                    return Add(*map(as_egraph, expr.args))
                case "float32":
                    return CastF32(*map(as_egraph, expr.args))

                case _:
                    raise NotImplementedError(expr.callee.name)
        case FloatLiteral() as expr:
            return Term.lit_f64(expr.fval)
        case IntLiteral() as expr:
            return Term.lit_i64(expr.ival)
        case Symbol() as expr:
            return Term.var(expr.name)
        case _:
            raise NotImplementedError(type(expr))
