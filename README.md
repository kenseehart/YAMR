# YAMR
 Yet Another Max Reduce (for CUDA)

This approach appears naive but isn't. This won't generalize to other functions like `sum()`, but it works great for `min()` and `max()`. Also, this implementation is for arrays up to one block (1024, 2048, 4096 depending on your hardware).

Yup, that's right, only syncing once after initialization and once before writing the result. Damn the race conditions! Full speed ahead!

It works significantly faster than a conventional reduction. Another surprise is that the average number of passes for a kernel size of 32 is 4. Yup, that's (log(n)-1), which seems counterintuitive. 

For terseness I've omitted the size and output arguments. Size is simply the number of threads (which could be 137 or whatever you like). Output is returned in `d_data[0]`.

## Theory

The magic is in the way the race condition plays out in our favor rather than causing problems. The race condition threatens to overwrite `max_value` with a smaller value than the one just written. If that happens, the while loop continues. But each time that happens `max_value` increases anyway. The while loop can't exit for the warp until no `v` in warp is larger than `max_value`. The first warp will take only 4 iterations in the while loop instead of the expected 5 because a lucky hit can cause an iteration to be skipped (e.g. the maximum might get written in the first pass). The possible lucky hits add up to an average saving of one pass. The big savings is due to the lack of synchronization and reads from shared memory. Writes are also very sparse after max_value grows higher in the distribution, as most warps exit after only one parallel compare.

With larger n, there is no way to avoid at least one iteration per warp, but that iteration typically only involves one compare operation which is usually immediately false across the entire warp when max_value is on the high end of the distribution. You could use a grid to use multiple SM's, but that would greatly increase the total workload and add a communication cost, so not likely to help.

## Race conditions

There are two theoretical race conditions to consider:

1. Multiple writes happening at the same time. This will happen a lot. Only one will land (this in guranteed by the spec), but which one is undefined. It turns out that this race condition doesn't matter because if a write is overwritten by another write of a smaller value, the while loop resolves it.

2. A write happens in one half-warp immediately after the compare, so the while loop exits prematurely. This one can't happen in practice (on NVidia hardware at least) because there is nothing to knock the two half-warps out of phase. In other words, the phase relationship for this race condition does not exist.

