# 2D Wave Equation
We implemented 2D Wave Equation by serial code and CUDA parallel code by C Programming.

## How to run code 

You could change settings such as dx, dy, dt, GridSize, BlockSize in the file directly.

* Serial code :

```
  gcc 2dwave.c -o 2dwave -lm
  !./2dwave
```

* CUDA code:

```
  !nvcc -o 2d_wave_cuda 2d_wave_cuda.cu
  !./2d_wave_cuda
```

##


### Kaggle Version

```
https://www.kaggle.com/code/cranky1104/2d-wave-equation
```
