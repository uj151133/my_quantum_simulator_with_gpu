from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'bit',
        ['models/bit.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            '/usr/local/include',  # Adjust according to your setup
            '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1',  # Adjust according to your setup
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-stdlib=libc++', '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'],
        extra_link_args=['-stdlib=libc++'],
    ),
]

setup(
    name='bit',
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='A test project using pybind11 and c++11',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
