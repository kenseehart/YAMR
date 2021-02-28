
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

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


__global__ void maxReduce(float* d_data)
{
    // compute max over all threads, store max in d_data[0]
    int i = threadIdx.x;
    __shared__ float max_value;

    if (i == 0) max_value = 0.0f;
    __syncthreads();

    float v = d_data[i];

    while (max_value < v)
    {
        max_value = v;
    }

    __syncthreads();
    if (i == 0) d_data[0] = max_value;
}


void testMax(int n)
{
    float* h_data, * d_data;
    float cpu_max = 0.0f;

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
    printf("cpu_max = %f, gpu_max = %f, result = %s\n", cpu_max, h_data[0], (cpu_max == h_data[0]) ? "PASS" : "FAIL");
}


int main()
{

    for (int i = 100; i < 150; i++)
    {
        testMax(i * 5);
    }

    return 0;
}
