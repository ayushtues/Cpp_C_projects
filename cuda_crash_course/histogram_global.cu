#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using std::accumulate;
using std::cout;
using std::generate;
using std::ios;
using std::ofstream;
using std::vector;

// Number of bins for our plots
constexpr int BINS = 7;
constexpr int DIV = ((26 + BINS -1 )/ BINS);


__global__ void histogram(char *a, int * result, int N)
{   // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int alpha_position;

    for(int i = tid; i < N; i += (gridDim.x * blockDim.x))
    {
        // Calculate the postion in the alphabet
        alpha_position = a[i] - 'a';
        atomicAdd(&result[alpha_position / DIV], 1);
    }
}


int main()
{   // Declare our problem size
    int N = 1<<24;

    // Allocate memory on the host
    vector<char> h_input(N);

    // Allocate memory for the binned results
    vector<int> h_result(BINS);

    // Initialize the array
    srand(1);
    generate(begin(h_input), end(h_input), [](){ return 'a' + (rand()%26);});

    // Allocate device memory
    char *d_input;
    int *d_result;
    cudaMalloc(&d_input, N);
    cudaMalloc(&d_result, BINS * sizeof(int));

    cudaMemcpy(d_input, h_input.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result.data(), BINS * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate number of threads/threadblock and number of thread blocks
    int THREADS = 512;
    int BLOCKS =  N / THREADS;

    // Launch the kernel
    histogram<<<BLOCKS, THREADS>>>(d_input, d_result, N);

    // Copy the result back
    cudaMemcpy(h_result.data(), d_result, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Functional test
    assert(N == accumulate(begin(h_result), end(h_result), 0));

    cout<<"COMPLETED SUCCESSFULLY\n";

    cudaFree(d_input);
    cudaFree(d_result);

}