# mypy: disable-error-code="empty-body"

from __future__ import annotations

from pprint import pprint
from dataclasses import dataclass
import json
from egglog import (
    EGraph,
    rewrite,
    ruleset,
    i64Like,
    function,
    i64,
    String,
)
from symarray.egraph import Term, Mul
from symarray.extract import Extraction, Node


@function
def Pow(x: Term, y: Term) -> Term: ...


@function
def PowConst(x: Term, i: i64Like) -> Term: ...


@ruleset
def expand_power(x: Term, y: Term, term: Term, i: i64, j: i64, k: String):
    yield rewrite(Pow(x, Term.lit_i64(i))).to(PowConst(x, i))
    yield rewrite(PowConst(x, 0)).to(Term.lit_f32(0.0))
    yield rewrite(PowConst(x, 1)).to(x)
    yield rewrite(PowConst(x, i)).to(Mul(x, PowConst(x, i - 1)), i > 1)

# Expression x ** 4
# Expected: x * x * x * x
rootexpr = Pow(Term.var("x"), Term.lit_i64(4))
egraph = EGraph()
egraph.let("root", rootexpr)
egraph.run((expand_power).saturate())
out, cost = egraph.extract(rootexpr, include_cost=True)

print(out)
print("cost", cost)

rawdata = egraph._serialize(n_inline_leaves=0).to_json()
data = json.loads(rawdata)
pprint(data)


class CostModel:
    def get_cost_function(
        self,
        nodename: str,
        op: str,
        nodes: dict[str, Node],
        child_costs: list[float],
    ) -> float:
        match op:
            case "Term.var" | "Term.lit_i64":
                return 0
            case "PowConst" | "Pow":
                return sum([100, *child_costs])
            case "Mul":
                return sum([2, *child_costs])
            case _:
                raise NotImplementedError(op)


ext = Extraction(data, root_eclass="Term-0", cost_model=CostModel())
cost, chosen = ext.choose()
print(f"cost={cost}")
ext.draw_graph(ext.nxg, "full.svg")
ext.draw_graph(chosen, "chosen.svg")
