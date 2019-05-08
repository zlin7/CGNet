from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_NAME= "cudaCG_dev"

import os
os.system("pip uninstall %s"%PACKAGE_NAME)

setup(
    name=PACKAGE_NAME,
    ext_modules=[
        CUDAExtension('CG_cuda', [
            'cudaCG.cpp',
            'cudaCG_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
