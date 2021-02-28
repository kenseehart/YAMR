# YAMR
 Yet Another Max Reduce (for CUDA)

This approach appears naive but isn't. This won't generalize to other functions like `sum()`, but it works great for `min()` and `max()`.

    __global__ void maxKernel(float* d_data)
    { 
        // compute max over all threads, store max in d_data[0]
        int i = threadIdx.x;
        __shared__ float max_value;
    
        if (i == 0) max_value = 0.0f;
        __syncthreads();
    
        float v = d_data[i];
    
        while (max_value < v) max_value = v;
    
        __syncthreads();
        if (i == 0) d_data[0] = max_value;
    }

Yup, that's right, only syncing once after initialization and once before writing the result. Damn the race conditions! Full speed ahead!

It turns out that the race condition doesn't matter in this case because the while loop resolves it.

It works significantly faster than a conventional reduction. Another surprise is that the average number of passes for a kernel size of 32 is 4. Yup, that's (log(n)-1), which seems counterintuitive. It's because the race condition gives an opportunity for good luck. This bonus comes in addition to removing the overhead of the conventional reduction.

With larger n, there is no way to avoid at least one iteration per warp, but that iteration only involves one compare operation which is usually immediately false across the warp when max_value is on the high end of the distribution. You could modify it to use multiple SM's, but that would greatly increase the total workload and add a communication cost, so not likely to help.

For terseness I've omitted the size and output arguments. Size is simply the number of threads (which could be 137 or whatever you like). Output is returned in `d_data[0]`.
