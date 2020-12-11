#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

// define matrix and shared memory size
const int N = 1<<10;
const int SHMEM_SIZE = 32 * 32 * 4;


__global__ void matrixMul(const int* a, const int* b, int* c)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocate  shared memory tiles of size BLOCK_SIZE * BLOCK_SIZE * Sizeof(Int)
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    // to store the accumulated sum
    int temp = 0;

    // sweep tile accross the matrix
    for(int i = 0 ; i < N; i+= blockDim.x)
    {   
        // Load elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x ] = a[ row*N + i + threadIdx.x];
        s_b[threadIdx.y*blockDim.x + threadIdx.x] = b[i*N + threadIdx.y*N + col];

        // Wait for both tiles to be loaded in before during computation
        __syncthreads();

        for(int j = 0; j < blockDim.x; j++)
        {
            temp += s_a[threadIdx.y * blockDim.x + j] * s_b[j*blockDim.x + threadIdx.x];
        }        

        // Wait for all threads to finish using current tiles before loading in new ones
        __syncthreads();

    }

    c[row * N + col] = temp;
    
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main()
{
    // size of the matrix in bytes
    size_t bytes = N * N * sizeof(int);

    //declare the matrices
    vector <int> h_a(N * N);
    vector <int> h_b(N * N);
    vector <int> h_c(N * N);

    // Initialize matrices
    generate(h_a.begin(), h_a.end(), [](){ return rand() % 100;});
    generate(h_b.begin(), h_b.end(), [](){ return rand() % 100;});

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // send data to GPU
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // number of threads in a block
    int THREADS = 32;

    // number of blocks in a grid
    int BLOCKS = N / THREADS;

    // use dim3 to allocate 2d blocks and grids
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

    // copy data back from GPU
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // check result
    verify_result(h_a, h_b, h_c);

    cout<< "COMPLETED SUCCESSFULLY!\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}