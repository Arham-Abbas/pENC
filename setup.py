from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="inverse_wave",
        sources=["inverse_wave.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="inverse_wave_module",
    version="1.0",
    author="Arham Abbas",
    description="A module for inverting audio waves using Cython",
    ext_modules=cythonize(extensions)
)
