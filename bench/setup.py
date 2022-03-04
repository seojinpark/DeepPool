from os import link
from xml.etree.ElementInclude import include
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='deeppool_bench',
      ext_modules=[cpp_extension.CppExtension(
          name='deeppool_bench', sources=['bench.cpp'], extra_compile_args=['-g', '-I/root/miniconda3/lib/python3.9/site-packages/torch/include', '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64', '-std=c++17', '-fpermissive']
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
