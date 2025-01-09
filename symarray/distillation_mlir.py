from __future__ import annotations

import numpy as np
import ast
from typing import Any
from textwrap import indent
from .egraph import Term
from pprint import pprint
from . import syms, array, distillation

_template = """
def func({argspec}):
{body}
"""


def distill_to_mlir(tree: Term, argspec: str) -> str:
    expr = distillation.distill_to_expr(tree)
    argnames = map(lambda x: x.strip(), argspec.split(','))
    argmap = {k: f"%arg_{k}" for k in argnames}
    source = MLIRGen(expr, argmap).generate()
    return source


_prologue = r'''
func.func @do_work(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_0 : memref<?xf32>
    affine.for %arg2 = %c0 to %dim {
'''

_epilogue = r'''
    }
    return
}
'''
class MLIRGen:
    root: syms.Expr

    def __init__(self, root: syms.Expr, argmap: dict[str, str]):
        self.root = root
        self.cache = {}
        self.vars = ["a"]
        self.subexprs = {}

    def generate(self):
        subexprs = list(self.find_subexpr(self.root))
        subexprs.sort(key=lambda x: len(str(x)))

        buf = []
        for i, subex in enumerate(subexprs):
            self.recur_generate(subex)
            orig = self.cache[subex]
            k = f"%v{i}"
            self.cache[subex] = k
            self.subexprs[k] = orig
            buf.append(f"{k} = {orig}")

        self.recur_generate(self.root)
        res = self.cache[self.root]

        buf.append(f"affine.store {res}, %arg1[%arg2] : memref<?xf32>")
        output = _prologue + indent('\n'.join(buf), '    ' * 2) + _epilogue
        return output


    def find_subexpr(self, expr: syms.Expr):
        seen = set()
        subexprs = set()
        buffer = [expr]
        while buffer:
            tos = buffer.pop()
            subexprs.add(tos)
            if tos in seen:
                continue
            seen.add(tos)
            buffer.extend(get_children(tos))

        return subexprs

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
                            return f"math.sqrt {arg} : f32"
                        case "Mul":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"arith.mulf {lhs}, {rhs} : f32"
                        case "Div":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"arith.divf {lhs}, {rhs} : f32"
                        case "Add":
                            [lhs, rhs] = map(lookup, expr.args)
                            return f"arith.addf {lhs}, {rhs} : f32"
                        case "float32":
                            [arg] = map(lookup, expr.args)
                            return f"arith.sitofp {arg} : i32 to f32"

                        case _:
                            raise NotImplementedError(expr.callee.name)
                case syms.FloatLiteral() as expr:
                    return f"arith.constant {expr.fval:e} : f32"
                case syms.IntLiteral() as expr:
                    return f"arith.constant {expr.ival} : i32"
                case syms.Symbol(str(name)) as expr:
                    assert name in self.vars  # XXX
                    return f"affine.load %arg0[%arg2] : memref<?xf32>"
                case _:
                    raise NotImplementedError(type(expr))

        self.cache[expr] = as_source(expr)



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
