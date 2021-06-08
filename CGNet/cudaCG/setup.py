from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME= "cudaCG_all"

import os
os.system("pip uninstall %s"%PACKAGE_NAME)

setup(
    name=PACKAGE_NAME,
    ext_modules=[
        CUDAExtension('CG_cuda_ops', [
            'cudaCG_all.cpp',
            'cudaCG_utils.cu',
            #'cudaCG_no_filter.cu',
            #'cudaCG_no_filter_with_BM.cu',
            'cudaCG_sparse.cu',
            'cudaCG_FNWMM.cu'
        ],
                      extra_compile_args=[]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

#python setup.py install
#import CG_cuda_ops
#For some reason, we need to load pytorch first before loding i