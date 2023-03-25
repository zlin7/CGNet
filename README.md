# Clebsch-Gordan Nets: A Fully Fourier Space Spherical CNN

Code used for experiments in the paper "Clebsch–Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network", Risi Kondor, Zhen Lin, Shubhendu Trivedi, Advances in Neural Information Processing Systems, 2018
(link: https://papers.nips.cc/paper/8215-clebschgordan-nets-a-fully-fourier-space-spherical-convolutional-neural-network)

Update: Additionally, we also provide an implementation of some of the optimizations suggested in the paper: "Efficient Generalized Spherical CNNs" (ICLR 2021) by Cobb et al. https://arxiv.org/abs/2010.11661 These are optional and can be used for faster spherical CNNs (over a baseline CGNet) with some approximation. The corresponding code can be found labelled as "sparse" and "MST" in the python and CUDA code snippets. 


# License
Copyright (c) <2018> <Zhen Lin, Shubhendu Trivedi, Risi Kondor>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
with the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# Requirements:
python 3.8 (tested on 3.8.5)

pytorch 1.8.0

tested using cuda 11.1 (only because RTX 30 GPU was used for experiments. Older versions might work as well)

lie_learn: https://github.com/AMLab-Amsterdam/lie_learn 
Note that for windows system, you need to change `setup.py` in the main directory. 
Specifically, change `extensions = cythonize(extensions)` to `extensions = cythonize(files)`


## Getting Started

To be able to run Clebsch-Gordan nets on GPU, we need to build the relevant cuda code first. 

There are two versions of SphericalCNN. They are called SphericalCNN and SphericalCNN_fast for now. SphericalCNN has most operations coded up in Python so it is more readable. SphericalCNN_fast puts most of the operations in cuda kernels, so it is a little faster - they do the same thing when used with the same parameters in the paper, however.


### SphericalCNN

This module uses cudaCG module for the GPU support. To build this, (and perform quick testing) do the following:

```
cd CGNet/cudaCG
python setup.py install
```

Then run main.py with settings. See main.py for options and how to set them.

NOTE: although the package ClebschGordan is not necessary to run the SphericalCNN codes, it is used
in testing the cuda kernel. To install, perform "python setup_cextension.py install" in CGNet/ClebschGordan

## Application Example

### MNIST Example
see `MNIST/README.md`.

###SHREC17 example:
The original data seems removed. This part of the code is thus not updated yet.
