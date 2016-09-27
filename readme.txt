To run the mm-cuda.cu, just use the same command as stated in Assignment

nvcc ¨Carch=sm_32 mm-cuda.cu ¨Co mm-cuda ¨Clcuda -lcudart

And when running, the program would give you three time, 

The first running time is for the cpu.
The second running time is for the gpu after optimization.
The third one is for the original gpu method.

And my program would check the value of these three matrix to see whether they are the same.


