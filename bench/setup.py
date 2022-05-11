from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='deeppool_bench',
      ext_modules=[CUDAExtension(
          name='deeppool_bench', sources=['bench.cpp'], extra_compile_args=['-g','-I/usr/local/cuda/include','-L/usr/local/cuda/lib64']
      )],
      cmdclass={'build_ext': BuildExtension})
