from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import numpy as np

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith("win"):
    extra_compile_args += ["/O2", "/std:c++17", "/EHsc", "/DNDEBUG"]
else:
    extra_compile_args += ["-O3", "-std=c++17", "-DNDEBUG"]

ext_modules = [
    Pybind11Extension(
        "cpoker",
        ["poker_env.cpp"],              # <- arquivo C++ atual (DeepCFR)
        include_dirs=[np.get_include(), "."],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    )
]

setup(
    name="cpoker",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
