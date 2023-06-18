import os
from distutils.command.build_ext import build_ext as build_ext_distutils
from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup


class build_ext(build_ext_distutils):
    def get_export_symbols(self, ext):
        return ext.export_symbols


def get_extension():
    mkl_root = Path(os.environ["MKLROOT"])
    mkl_include_dirs = str(mkl_root / "include")
    mkl_library_dirs = str(mkl_root / "lib" / "intel64")
    mkl_libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core"]

    return Extension(
        "mkl_bind/_mkl_bind",
        ["mkl_bind/_mkl_bind.c"],
        include_dirs=[np.get_include()] + [mkl_include_dirs],
        libraries=mkl_libraries,
        library_dirs=[mkl_library_dirs],
        extra_compile_args=["-DNDEBUG"],
    )


setup(
    name="mkl-bind",
    version="0.0.1",
    description="mkl-bind provides a minimalistic Python binding to the Intel MKL FFT",
    author="Styfen Schär",
    author_email="styfen.schaer.blog@gmail.com",
    url="https://github.com/styfenschaer/mkl-bind",
    download_url="https://github.com/styfenschaer/mkl-bind",
    packages=find_packages(),
    include_package_data=True,
    package_data={"mkl_bind": ["*.pyi"]},
    ext_modules=[get_extension()],
    cmdclass={
        "build_ext": build_ext,
    },
)
