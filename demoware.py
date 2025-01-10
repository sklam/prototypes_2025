import os
os.environ["LLVMLITE_ENABLE_OPAQUE_POINTERS"] = "1"
import numpy as np
import types as pytypes
import ctypes
import ctypes.util
from llvmlite import binding as llvm
from tempfile import NamedTemporaryFile
import subprocess
from functools import cache
import sys

@cache
def init_llvm():
    llvm.initialize()
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_mod(engine, mod):
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


def as_memref_descriptor(arr, ty):
    intptr_t = getattr(ctypes, f"c_int{8 * ctypes.sizeof(ctypes.c_void_p)}")
    N = arr.ndim

    ty_ptr = ctypes.POINTER(ty)
    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ty_ptr),
            ("aligned", ty_ptr),
            ("offset", intptr_t),
            ("sizes", intptr_t * N),
            ("strides", intptr_t * N),
        ]

    arg0 = ctypes.cast(arr.ctypes.data, ty_ptr)
    arg1 = arg0
    arg2 = intptr_t(0)
    arg3 = (intptr_t * N)(*arr.shape)
    arg4 = (intptr_t * N)(*arr.strides)
    return MemRefDescriptor(arg0, arg1, arg2, arg3, arg4)


class MLIRCompiler():
    def __init__(self, debug=False, print_cmds=False):
        self._debug = debug
        self._print_cmds = print_cmds

    def _run_cmd(self, cmd, in_mode, out_mode, src):
        assert in_mode in "tb"
        assert out_mode in "tb"
        with NamedTemporaryFile(mode=f"w{in_mode}") as src_file:
            src_file.write(src)
            src_file.flush()

            with NamedTemporaryFile(mode=f"r{out_mode}") as dst_file:
                full_cmd = *cmd, src_file.name, "-o", dst_file.name
                if self._print_cmds:
                    print(full_cmd)
                subprocess.run(full_cmd)
                dst_file.flush()
                return dst_file.read()

    def to_llvm_dialect_with_omp_target(self, mlir_src):
        # mlir source to mlir llvm dialect source transform
        binary = ("mlir-opt",)
        if self._debug:
            dbg_cmd = ("--mlir-print-debuginfo",
                       "--mlir-print-ir-after-all",
                       "--debug-pass=Details",)
        else:
            dbg_cmd = ()

        options = (
            "--debugify-level=locations",
            "--use-unknown-locations=Enable",
            "--snapshot-op-locations",
            "--experimental-debug-variable-locations",
            "--dwarf-version=5",
            "--inline",
            "-affine-loop-normalize",
            "-affine-parallelize",
            "-affine-super-vectorize",
            "--affine-scalrep",
            "-lower-affine",
            "-convert-vector-to-scf",
            "-convert-linalg-to-loops",
            "-lower-affine",
            "-convert-scf-to-openmp",
            "-convert-scf-to-cf",
            "-cse",
            "-convert-openmp-to-llvm",
            "-convert-linalg-to-llvm",
            "-convert-vector-to-llvm",
            "-convert-math-to-llvm",
            "-expand-strided-metadata",
            "-lower-affine",
            "-finalize-memref-to-llvm",
            "-convert-func-to-llvm",
            "-convert-index-to-llvm",
            "-reconcile-unrealized-casts",
            "--llvm-request-c-wrappers",
        )
        full_cmd = binary + dbg_cmd + options
        return self._run_cmd(full_cmd, "t", "t", mlir_src)

    def mlir_translate_to_llvm_ir(self, mlir_src):
        # converts mlir source to llvm ir source
        binary = ("mlir-translate",)
        options = (
            "--mlir-print-local-scope",
            "--mir-debug-loc",
            "--use-unknown-locations=Enable",
            "-mlir-print-debuginfo=true",
            "--experimental-debug-variable-locations",
            "--mlir-to-llvmir",
        )
        full_cmd = binary + options
        return self._run_cmd(full_cmd, "t", "t", mlir_src)

    def llvm_ir_to_bitcode(self, llvmir_src):
        # converts llvm ir source to llvm bitcode
        binary = ("llvm-as",)
        full_cmd = binary
        return self._run_cmd(full_cmd, "t", "b", llvmir_src) # txt to binary


class Compiler():
    def __init__(self):
        init_llvm()
        # Need to load in OMP into process for the OMP backend.
        if sys.platform.startswith('linux'):
            omppath = ctypes.util.find_library("libgomp.so")
        elif sys.platform.startswith('darwin'):
            omppath = ctypes.util.find_library("iomp5")
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        ctypes.CDLL(omppath, mode=os.RTLD_NOW)

        self.ee = create_execution_engine()

    def run_frontend(self, py_func, sig):
        fname = "testme.mlir"
        with open(fname, "rt") as f:
            mlir_src = f.read()
        return mlir_src

    def run_backend(self, mlir_src):
        mlir_compiler = MLIRCompiler(debug=False, print_cmds=False)
        mlir_omp = mlir_compiler.to_llvm_dialect_with_omp_target(mlir_src)
        llvm_ir = mlir_compiler.mlir_translate_to_llvm_ir(mlir_omp)
        fdata = mlir_compiler.llvm_ir_to_bitcode(llvm_ir)
        mod = llvm.parse_bitcode(fdata)
        mod = compile_mod(self.ee, mod)

        address = self.ee.get_function_address("_mlir_ciface_do_work")
        assert address, "Lookup for compiled function address is returning NULL."
        return address

    def jit_compile(self, py_func, sig):
        mlir = self.run_frontend(py_func, sig)
        address = self.run_backend(mlir)
        return address


class Dispatcher():
    def __init__(self, py_func):
        self.py_func = py_func
        self._compiled_func = None
        self._compiler = None

    def compile(self, sig):
        self._compiler = Compiler()
        binary = self._compiler.jit_compile(self.py_func, sig)
        self._compiled_func = binary
        return binary

    def __call__(self, *args, **kwargs):
        assert not kwargs
        if self._compiled_func is None:
            self.compile("TODO: work out signature from types in *args")

        out = np.empty_like(args[0])
        all_args = *args, out
        args_as_memref = [as_memref_descriptor(x, ctypes.c_double) for x in all_args]
        prototype = ctypes.CFUNCTYPE(None, *[ctypes.POINTER(type(x)) for x in args_as_memref])
        cfunc = prototype(self._compiled_func)
        cfunc(*[ctypes.byref(x) for x in args_as_memref])
        return out


def vectorize(func_or_sig):
    if isinstance(func_or_sig, pytypes.FunctionType):
        wrap = Dispatcher(func_or_sig)
    elif isinstance(func_or_sig, str): # it's a signature
        def wrap(py_func):
            disp = Dispatcher(py_func)
            disp.compile(func_or_sig)
            return disp
    else:
        raise TypeError("Expected a python function or a string signature")
    return wrap

