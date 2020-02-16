#!/bin/bash

TENSORFLOW_CORE=/tensorflow-2.1.0/python3.6/tensorflow_core
OPS_FILE=tf_ops
CUDA_ROOT=/usr/local/cuda-10.1

g++ -std=c++11 $OPS_FILE/3d_interpolation/tf_interpolate.cpp -o $OPS_FILE/tf_interpolate_so.so -shared -fPIC \
-I$TENSORFLOW_CORE/include -I$CUDA_ROOT/include -I$TENSORFLOW_CORE/include/external/nsync/public -lcudart -L$CUDA_ROOT/lib64/ \
-L$TENSORFLOW_CORE -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

nvcc $OPS_FILE/grouping/tf_grouping_g.cu -o $OPS_FILE/grouping/tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC &&\
g++ -std=c++11 $OPS_FILE/grouping/tf_grouping_g.cu.o $OPS_FILE/grouping/tf_grouping.cpp -o $OPS_FILE/tf_grouping_so.so -shared -fPIC \
-I$TENSORFLOW_CORE/include -I$CUDA_ROOT/include -I$TENSORFLOW_CORE/include/external/nsync/public -lcudart -L$CUDA_ROOT/lib64/ \
-L$TENSORFLOW_CORE -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

nvcc $OPS_FILE/sampling/tf_sampling_g.cu -o $OPS_FILE/sampling/tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC &&\
g++ -std=c++11 $OPS_FILE/sampling/tf_sampling_g.cu.o $OPS_FILE/sampling/tf_sampling.cpp -o $OPS_FILE/tf_sampling_so.so -shared -fPIC \
-I$TENSORFLOW_CORE/include -I$CUDA_ROOT/include -I$TENSORFLOW_CORE/include/external/nsync/public -lcudart -L$CUDA_ROOT/lib64/ \
-L$TENSORFLOW_CORE -l:libtensorflow_framework.so.2 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
