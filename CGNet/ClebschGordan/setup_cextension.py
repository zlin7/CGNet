#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('ClebschGordan', sources = ['CGmodule.cpp', "ClebschGordan.cpp"]) ]


PACKAGE_NAME= "ClebschGordan"

import os
os.system("pip uninstall %s"%PACKAGE_NAME)


setup(
        name = PACKAGE_NAME,
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )

