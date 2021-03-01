# YAMR
 Yet Another Max Reduce (for CUDA)

This approach appears naive but isn't. This won't generalize to other functions like `sum()`, but it works great for `min()` and `max()`.

    __device__ const float float_min = -3.402e+38;
    
    __global__ void maxKernel(float* d_data)
    { 
        // compute max over all threads, store max in d_data[0]
        int i = threadIdx.x;
        __shared__ float max_value;
    
        if (i == 0) max_value = float_min;
    
        float v = d_data[i];
        __syncthreads();
    
        while (max_value < v) max_value = v;
    
        __syncthreads();
        if (i == 0) d_data[0] = max_value;
    }

Yup, that's right, only syncing once after initialization and once before writing the result. Damn the race conditions! Full speed ahead!

It works significantly faster than a conventional reduction. Another surprise is that the average number of passes for a kernel size of 32 is 4. Yup, that's (log(n)-1), which seems counterintuitive. It's because the race condition gives an opportunity for good luck. This bonus comes in addition to removing the overhead of the conventional reduction.

With larger n, there is no way to avoid at least one iteration per warp, but that iteration only involves one compare operation which is usually immediately false across the warp when max_value is on the high end of the distribution. You could modify it to use multiple SM's, but that would greatly increase the total workload and add a communication cost, so not likely to help.

For terseness I've omitted the size and output arguments. Size is simply the number of threads (which could be 137 or whatever you like). Output is returned in `d_data[0]`.

## Theory

The magic is in the way the race condition plays out in our favor rather than causing problems. The race condition threatens to overwrite `max_value` with a smaller value than the one just written. If that happens, the while loop continues. But each time that happens `max_value` increases anyway. The while loop can't exit for the warp until no `v` in warp is larger than `max_value`. The first warp will take only 4 iterations in the while loop instead of the expected 5 because a lucky hit can cause an iteration to be skipped (e.g. the maximum might get written in the first pass). The possible lucky hits add up to an average saving of one pass. The big savings is due to the lack of synchronization and reads from shared memory. Writes are also very sparse after max_value grows higher in the distribution, as most warps exit after only one parallel compare.

## Race conditions

There are two theoretical race conditions to consider:

1. Multiple writes happening at the same time. This will happen a lot. Only one will land (this in guranteed by the spec), but which one is undefined. It turns out that this race condition doesn't matter because if a write is overwritten by another write of a smaller value, the while loop resolves it.

2. A write happens in one half-warp immediately after the compare, so the while loop exits prematurely. This one can't happen in practice (on NVidia hardware at least) because there is nothing to trigger a swap between warps (e.g. IO latency condition) and there is nothing to knock the two half-warps out of phase. Note that if there is any non-register memory access between the `__syncthread();` operations, all bets are off. If you are not comfortable with this, you can use `AtomicMax()` with a small reduction in performance.

