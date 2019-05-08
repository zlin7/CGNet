# Clebsch-Gordan Nets: A Fully Fourier Space Spherical CNN

Code used for experiments in the paper "Clebsch–Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network", Risi Kondor, Zhen Lin, Shubhendu Trivedi, Advances in Neural Information Processing Systems, 2018
(link: https://papers.nips.cc/paper/8215-clebschgordan-nets-a-fully-fourier-space-spherical-convolutional-neural-network)

# License
Copyright (c) <2018> <Zhen Lin, Shubhendu Trivedi, Risi Kondor>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
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


# Requirement:
python 3.6 (tested on 3.6.8)

pytorch 1.1.0

tested using cuda 10.0

lie_learn: https://github.com/AMLab-Amsterdam/lie_learn

## Getting Started

To be able to run Clebsch-Gordan nets on GPU, we need to build the relevant cuda code first. 

There are two versions of SphericalCNN. They are called SphericalCNN and SphericalCNN_fast for now. SphericalCNN has most operations coded up in Python so it is more readable. SphericalCNN_fast puts most of the operations in cuda kernels, so it is a little faster - they do the same thing when used with the same parameters in the paper, however.


### SphericalCNN

This module uses cudaCG module for the GPU support. To build this, (and perform quick testing) do the following:

```
cd CGNet/cudaCG/cuda
./run.sh
```

Then you can run main.py with settings. See main.py for options and how to set them.

NOTE: although the package ClebschGordan is not necessary to run the SphericalCNN codes, it is used
in testing the cuda kernel. To install, perform "python setup_cextension.py install" in CGNet/ClebschGordan

### SphericalCNN_fast

This module has almost everything on GPU, not just the CG decomposition. To build its GPU support, do the following:

```
cd CGNet/cudaUpBatchNorm
./run.sh
```

Then you can run main_fast.py with settings. See main_fast.py for options and how to set them.


## Application Example

### MNIST Example

Run the following command to precompute the data for MNIST training and testing.

```
python datautils.py all
```

Then, if both SphericalCNN modules are correctly set up, 

```
bash example.sh
```

will run something. Logs can be found in "temp_new" folder.


###Shreac17 example
This is yet to be tested on python3 + pytorch 1.0. 