#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#define SHMEM_SIZE 256

__global__ void sum_reduction(int *v, int *v_r)
{   
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_SIZE];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * ( 2 * blockDim.x) + threadIdx.x;

    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Iterate of logbase 2 the block dimension
    for(int s = blockDim.x/2; s > 0 ; s>>=1 )
    {
        if(threadIdx.x < s) // each thread does work unless it goes off the block
        {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s ];
        }

        __syncthreads();
    }

    // Let the thread 0 for this block write it's results to main memory
    // Result indexed by this block
    if(threadIdx.x == 0 )
    {
        v_r[blockIdx.x] = partial_sum[0];
    }

}

int main()
{   
    // vector size
    int N = 1<<16;
    size_t bytes = N * sizeof(int);

    // host data
    vector<int> h_v(N);
    vector<int> h_v_r(N);

    // initialize vector
    generate(begin(h_v), end(h_v), [](){return rand() % 10; });

    // device memory
    int *d_v, *d_v_r;

    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    // copy from host ( CPU ) to device ( GPU )
    cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);
    
    // Thread block size
    const int TB_SIZE = 256;

    // The Grid size
    int GRID_SIZE = N/TB_SIZE/2;

    // call the kernels
    sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r); // first convert the 65536 vector into a 256 sized vector of partial sums
    sum_reduction<<<1, TB_SIZE>>>(d_v_r, d_v_r); // use the 256 sized vector of partial sums to calculate the final sum

    cudaMemcpy(h_v_r.data(), d_v_r, bytes, cudaMemcpyDeviceToHost);

    // check the result
    assert(h_v_r[0] == std::accumulate(begin(h_v), end(h_v), 0));

    cout<<"COMPLETED SUCCESSFULLY\n";

    return 0;


}