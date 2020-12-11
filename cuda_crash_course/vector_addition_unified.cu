#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// __global__ means that its called from CPU to run on the GPU
__global__ void vectorAdd(int *a, int *b, int *c, int N)
{   
    // calculate the global thread id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tid < N) // a quick boundary check
    {
        c[tid]  = a[tid] + b[tid]; // do the actual addition
    }
}


int main()
{
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // declare the vectors to hold the data on the CPU 
    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    int id = cudaGetDevice(&id);

    // give some hints to the memory manager about where we want our variables to live
    cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, id);

    // fill up the vectors
    for(int i=0; i < N; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // prefetch a and b to GPU
    cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    // Threads per CTA
    int BLOCK_SIZE = 1<<10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int GRID_SIZE = (N + BLOCK_SIZE - 1)/ BLOCK_SIZE;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous ( the CPU program continues execution after 
    // call, but not necessarily before the kernel finishes)
    vectorAdd<<< GRID_SIZE, BLOCK_SIZE >>>(a, b, c, N);

    
    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    // prefetch back to CPU
    cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    for (int i = 0 ; i<N ; i++)
    {
        assert(c[i] == a[i] + b[i]);
    }

    // Free memory on device
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;



}

