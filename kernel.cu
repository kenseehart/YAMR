/*
MIT License

Copyright(c) 2021 kenseehart

Permission is hereby granted, free of charge, to any person obtaining a copy of
this softwareand associated documentation files(the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and /or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const float float_min = -3.402e+38;

__device__ const float d_float_min = -3.402e+38;

__host__ void _cudaCheckError(const char* file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fatal error: %s at %s[%d]\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "*** CUDA FAILED - ABORTING\n");
        exit(1);
    }
}

#define cudaCheckError _cudaCheckError(__FILE__, __LINE__)


__global__ void maxReduce(volatile float* d_data)
{
    // compute max over all threads, store max in d_data[0]
    int i = threadIdx.x;
    __shared__ volatile float max_value;

    if (i == 0) max_value = d_float_min;

    float v = d_data[i];
    __syncthreads();

    while (max_value < v)
    {
        max_value = v;
    }

    __syncthreads();
    if (i == 0) d_data[0] = max_value;
}


void testMax(int n, bool verbose)
{
    float* h_data, * d_data;
    float cpu_max = float_min;

    // Allocate memory mapped data
    cudaHostAlloc((void**)&h_data, n * sizeof(float), cudaHostAllocMapped); cudaCheckError;
    cudaHostGetDevicePointer((int**)&d_data, (int*)h_data, 0); cudaCheckError;

    for (int i = 0; i < n; i++)
    {
        // randomize
        h_data[i] = (float)rand() / (float)(1 + rand());

        // get cpu opinion of the max for testing
        if (cpu_max < h_data[i]) cpu_max = h_data[i];
    }

    // run the kernel
    maxReduce << <1, n >> > (d_data); cudaCheckError;
    cudaDeviceSynchronize(); cudaCheckError;

    // did the gpu get the same answer as the cpu?
    if (verbose)
    {
        printf("n =%6d cpu_max =%12.4f, gpu_max =%12.4f, result = %s\n", n, cpu_max, h_data[0], (cpu_max == h_data[0]) ? "PASS" : "FAIL");
    }
    else
    {
        if (cpu_max != h_data[0])
        printf("FAIL");
    }
}


int main()
{
    for (int j = 1; j < 5000; j++)
    {
        for (int i = 1; i < 16; i++)
        {
            testMax(i*31, j<20);
        }
    }
    return 0;
}
