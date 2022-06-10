from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "liftlab._cpp",
    ["cpp/src/bindings.cpp"],
    include_dirs=["cpp/include"],
    cxx_std=17,
    extra_compile_args=["-O3"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
