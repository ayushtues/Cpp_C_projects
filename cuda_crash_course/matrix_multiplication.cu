#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int* a, const int* b, int*c, int N)
{
    // computer threads global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col ] = 0;

    for(int k=0; k < N ; k++)
    {
        c[row * N + col] += a[row*N + k] * b[k*N + col]; // add a[row][k] * b[k][col]
    }
}


void verify_result(vector<int> &a, vector <int> &b, vector <int> &c, int N)
{
    for(int i = 0 ; i < N ; i++)
    {
        for(int j = 0; j < N ; j++)
        {
            int temp = 0;
            for(int k = 0 ; k < N ; k++)
            {
                temp += a[i * N + k] * b[k * N + j];
            }
            assert(temp == c[i * N + j]);
        }
    }
}


int main()
{
    // lets do 1024 X 1024 matrix multiplication
    int N = 1<<10;

    size_t bytes = N * N * sizeof(int);

    // allocate the host tensors
    vector <int> h_a(N * N);
    vector <int> h_b(N * N);
    vector <int> h_c(N * N);

    generate(h_a.begin(), h_a.end(), [](){ return rand() % 100; }); // A STL function to generate numbers and assign it to a container ( here vector )
    generate(h_b.begin(), h_b.end(), [](){ return rand() % 100; });

    int *d_a, *d_b, *d_c;

    // allocate memory on the GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // move tensors from host (CPU) to device (GPU)
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // threads per block
    int THREADS = 32;

    // number of blocks
    int BLOCKS = N / THREADS;

    // use dim3 object to store 2d threads and blocks
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // do the matrix multiplication on the gpu
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    // get data back from the gpu to the cpu
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}