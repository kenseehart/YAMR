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


__global__ void maxReduce(volatile float* d_data, int n)
{
    // compute max over all threads, store max in d_data[0]
    int ti = threadIdx.x;

    __shared__ volatile float max_value;

    if (ti == 0) max_value = d_float_min;

    for (int bi = 0; bi < n; bi += 32)
    {
        int i = bi + ti;
        if (i >= n) break;
        
        float v = d_data[i];
        __syncthreads();

        while (max_value < v)
        {
            max_value = v;
        }

        __syncthreads();
    }

    if (ti == 0) d_data[0] = max_value;
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
        h_data[i] = (float)rand() / (1 + rand()) - (float)rand() / (1 + rand());

        // get cpu opinion of the max for testing
        if (cpu_max < h_data[i]) cpu_max = h_data[i];
    }

    // run the kernel
    maxReduce << <1, 32 >> > (d_data, n); cudaCheckError;
    cudaDeviceSynchronize(); cudaCheckError;

    // did the gpu get the same answer as the cpu?
    if (verbose)
    {
        printf("n =%6d cpu_max =%12.4f, gpu_max =%12.4f, result = %s\n", n, cpu_max, h_data[0], (cpu_max == h_data[0]) ? "PASS" : "FAIL");
    }
    else
    {
        if (cpu_max != h_data[0])
        {
            printf("\nn =%6d %02dw+%02d cpu_max =%12.4f, gpu_max =%12.4f, result = FAIL\n", n, n/32, n%32, cpu_max, h_data[0]);
        }
    }
}


int main()
{
    for (int j = 1; j < 50000; j++)
    {
        int n = 2 + rand() % 10000;

        testMax(n, j<20);

        if (j > 20 && j%100==0) printf(".");
    }
    return 0;
}
