from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name='Video hasher',
  ext_modules=[
    Extension('frame_hash',
              sources=['frame_hash.pyx'],
              extra_compile_args=['-O3'],
              language='c++')
    ],
  cmdclass={'build_ext': build_ext}
)
