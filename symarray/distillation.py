from __future__ import annotations

import numpy as np
import ast
from typing import Any
from textwrap import indent
from .egraph import Term
from . import syms, array

_template = """
def func({argspec}):
{body}
"""


def distill_to_callable(tree: Term, argspec: str) -> str:
    source = SourceGen(distill_to_expr(tree)).generate()

    funcsrc = _template.format(argspec=argspec, body=indent(source, " " * 4))
    ns = {"sqrt": np.sqrt, "float32": np.float32}
    print(funcsrc)
    local_ns = {}
    exec(funcsrc, ns, local_ns)
    return local_ns["func"]


def distill_to_expr(tree: Term) -> syms.Expr:
    raw = str(tree)

    class Term:
        @classmethod
        def lit_f32(cls, v: float) -> syms.Expr:
            return syms.FloatLiteral(v)

        @classmethod
        def lit_f64(cls, v: float) -> syms.Expr:
            return syms.FloatLiteral(v)

        @classmethod
        def lit_i64(cls, v: int) -> syms.Expr:
            return syms.IntLiteral(v)

        @classmethod
        def var(cls, k: str) -> syms.Expr:
            return syms.Symbol(k)

    ns = {
        "Add": syms.Expr.__add__,
        "Mul": syms.Expr.__mul__,
        "Div": syms.Expr.__truediv__,
        "CastF32": array.float32,
        # "Tanh": array.tanh,
        "Sqrt": array.sqrt,
        "Term": Term,
    }

    astree = ast.parse(str(tree))
    astree.body[-1] = ast.fix_missing_locations(
        _ast_ensure_assignment(astree.body[-1])
    )
    local_ns: dict[str, Any] = {}
    exec(compile(astree, "<string>", "exec"), ns, local_ns)
    result = local_ns["__result__"]
    return result



def _ast_ensure_assignment(tree: ast.AST) -> ast.AST:
    match tree:
        case ast.Expr() as expr:
            return ast.Assign(
                targets=[ast.Name("__result__", ast.Store())],
                value=expr.value,
            )
    return tree


class SourceGen:
    root: syms.Expr

    def __init__(self, root: syms.Expr):
        self.root = root
        self.cache = {}
        self.subexprs = {}

    def generate(self):
        subexprs = list(self.find_subexpr(self.root))
        subexprs.sort(key=lambda x: len(str(x)))

        buf = []
        for i, subex in enumerate(subexprs):
            self.recur_generate(subex)
            orig = self.cache[subex]
            k = f"_Subex_{i}"
            self.cache[subex] = k
            self.subexprs[k] = orig
            buf.append(f"{k} = {orig}")

        self.recur_generate(self.root)
        res = self.cache[self.root]
        buf.append(f"return {res}")
        return "\n".join(buf)

    def recur_generate(self, expr: syms.Expr):
        if expr in self.cache:
            return

        def lookup(expr: syms.Expr) -> str:
            if expr in self.cache:
                return self.cache[expr]
            else:
                return as_source(expr)

        def as_source(expr: syms.Expr) -> str:
            match expr:
                case syms.CallExpr() as expr:
                    match expr.callee.name:
                        case "sqrt":
                            [arg] = map(lookup, expr.args)
                            return f"sqrt({arg})"
                        case "Mul":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"({lhs}) * ({rhs})"
                        case "Div":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"({lhs}) / ({rhs})"
                        case "Add":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"({lhs}) + ({rhs})"
                        case "float32":
                            [arg] = map(lookup, expr.args)
                            return f"float32({arg})"

                        case _:
                            raise NotImplementedError(expr.callee.name)
                case syms.FloatLiteral() as expr:
                    return f"float32({expr.fval})"
                case syms.IntLiteral() as expr:
                    return str(expr.ival)
                case syms.Symbol() as expr:
                    return expr.name
                case _:
                    raise NotImplementedError(type(expr))

        self.cache[expr] = as_source(expr)

    def find_subexpr(self, expr: syms.Expr):
        seen = set()
        subexprs = set()
        buffer = [expr]
        while buffer:
            tos = buffer.pop()
            if tos in seen:
                subexprs.add(tos)
                continue
            seen.add(tos)
            buffer.extend(get_children(tos))

        return subexprs


def get_children(expr: syms.Expr):
    match expr:
        case syms.CallExpr() as expr:
            return {*expr.args}
        case syms.FloatLiteral() as expr:
            return set()
        case syms.IntLiteral() as expr:
            return set()
        case syms.Symbol() as expr:
            return set()
        case _:
            raise NotImplementedError(type(expr))
