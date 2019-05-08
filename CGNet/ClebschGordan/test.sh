#!/bin/sh
# This is a comment!
#g++ -o check C2Python.cpp -I/usr/include/python2.7/ -lpython2.7
#python setup_cextension.py build_ext --inplace
python setup_cextension.py install
gdb `which python`
#python ./test.py