import symarray.array as np
from symarray.lift import lift
from symarray.egraph import as_egraph, Term
from symarray.distillation import distill_to_callable
from pprint import pprint
from egglog import eq, rewrite, rule, birewrite
from egglog import EGraph, ruleset, i64, f64
from egglog import converter


def geglu_tanh_forward_ufunc(a):
    dt = np.float32
    result = (
        dt(0.5)
        * a
        * (
            dt(1)
            + np.tanh(np.sqrt(dt(2) / dt(np.pi)) * (a + dt(0.044715) * a**3))
        )
    )
    return result


orig_func = geglu_tanh_forward_ufunc

exprtree = lift(orig_func)
pprint(exprtree)


eg_root = as_egraph(exprtree)

# ----------------------------------- Egraph -----------------------------------


@ruleset
def basic_math(x: Term, y: Term, z: Term, i: i64, fval: f64):

    from symarray.egraph import Add, Mul, Div, Pow, PowConst

    def one_way_birewrite(a, b):
        yield rewrite(a, subsume=True).to(b)
        yield rewrite(b).to(a)

    # Operator to Explicit calls
    # yield rewrite(x + y, subsume=True).to(Add(x, y))
    # yield rewrite(x * y, subsume=True).to(Mul(x, y))
    # yield rewrite(x / y, subsume=True).to(Div(x, y))
    # yield rewrite(x ** y, subsume=True).to(Pow(x, y))
    yield from one_way_birewrite(x + y, Add(x, y))
    yield from one_way_birewrite(x * y, Mul(x, y))
    yield from one_way_birewrite(x / y, Div(x, y))
    yield from one_way_birewrite(x**y, Pow(x, y))

    # Basic
    for binop in [Add, Mul]:
        # commutativity
        # yield rewrite(binop(x, binop(y, z))).to(binop(binop(y, z), x))
        # associativity
        yield rewrite(binop(x, binop(y, z))).to(binop(binop(x, y), z))

    # Pow
    yield rewrite(Pow(x, Term.lit_i64(i))).to(PowConst(x, i))
    yield rewrite(PowConst(x, 0)).to(Term.lit_f32(0.0))
    yield rewrite(PowConst(x, 1)).to(x)
    yield rewrite(PowConst(x, i)).to(Mul(x, PowConst(x, i - 1)), i > 1)


@ruleset
def const_propagation(x: Term, y: Term, z: Term, ival: i64, fval: f64):
    from symarray.egraph import CastF32, Div, Mul

    yield rewrite(CastF32(Term.lit_f64(fval))).to(Term.lit_f32(fval))
    # Div
    yield rewrite((x / Term.lit_f32(fval))).to(x * Term.lit_f32(1.0 / fval))


@ruleset
def pade44_tanh_expansion(x: Term, y: Term, z: Term):

    from symarray.egraph import Tanh

    flt = lambda f: Term.lit_f32(float(f))
    liti64 = Term.lit_i64
    yield rewrite(Tanh(x)).to(
        (flt(10) * x ** liti64(3) + flt(105) * x)
        / (x ** liti64(4) + flt(45) * x ** liti64(2) + flt(105))
    )


egraph = EGraph()
egraph.let("root", eg_root)


schedule = (basic_math | pade44_tanh_expansion | const_propagation).saturate()
# schedule = (pade44_tanh_expansion ).saturate()
# egraph.saturate(schedule)
egraph.run(schedule)
extracted = egraph.extract(eg_root)
print(extracted)
# egraph.display()

print("distillation".center(80, "="))
opt_func = distill_to_callable(extracted, "a")
print("distill")


import numpy as np

arr = np.linspace(-1, 1, 10000000, dtype=np.float32)

expected = orig_func(arr)
got = opt_func(arr)

np.testing.assert_allclose(expected, got, rtol=5e-6)