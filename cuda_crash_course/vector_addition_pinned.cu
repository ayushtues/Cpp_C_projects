#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// __global__ means that its called from CPU to run on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N)
{   
    // calculate the global thread id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( tid < N) // a quick boundary check
    {
        c[tid]  = a[tid] + b[tid]; // do the actual addition
    }
}

// check if the vector addition was performed correctly
void verify_result(std::vector<int> &a, std::vector<int> &b, 
                   std::vector<int>&c){

    for(int i =0; i<a.size(); i++){
        assert(c[i] == a[i] + b[i]);
    }
}

int main()
{
    constexpr int N = 1 << 16;
    constexpr size_t bytes = sizeof(int) * N;

    // declare the vectors to hold the data on the CPU 
    std::vector<int> a;
    a.reserve(N);

    std::vector<int> b;
    b.reserve(N);

    std::vector<int> c;
    c.reserve(N);

    // fill up the vectors
    for(int i=0; i < N; i++)
    {
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    // allocate space on the GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // transfer data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // Threads per CTA
    int NUM_THREADS = 1<<10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1)/ NUM_THREADS;

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous ( the CPU program continues execution after 
    // call, but not necessarily before the kernel finishes)
    vectorAdd<<< NUM_BLOCKS, NUM_THREADS >>>(d_a, d_b, d_c, N);
    
    // get the result back from the GPU
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(a, b, c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;



}

