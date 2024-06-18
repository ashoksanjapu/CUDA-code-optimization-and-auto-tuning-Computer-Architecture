#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void sumRedKernel0(float *d_vec, unsigned int n) {
    extern __shared__ float ds_partialSum[];
    unsigned int t = threadIdx.x;
    ds_partialSum[t] = d_vec[t];

    for (unsigned int stride = 1; stride < n; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0 && t + stride < n)
            ds_partialSum[t] += ds_partialSum[t + stride];
    }

    if (t == 0)
        d_vec[0] = ds_partialSum[0];
}

__global__ void sumRedKernel1(float *d_vec, unsigned int n) {
    extern __shared__ float ds_partialSum[];
    if (threadIdx.x < n)
        ds_partialSum[threadIdx.x] = d_vec[threadIdx.x];
    else
        ds_partialSum[threadIdx.x] = 0.0;

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride)
            ds_partialSum[threadIdx.x] += ds_partialSum[threadIdx.x + stride];
    }

    if (threadIdx.x == 0)
        d_vec[0] = ds_partialSum[0];
}

__global__ void sumRedKernel2(float *d_vec, unsigned int n) {
    extern __shared__ float ds_partialSum[];
    unsigned int t = threadIdx.x;
   
    // Each thread loads two elements from global memory to shared memory
    if (t * 2 < n) {
        ds_partialSum[t] = d_vec[t * 2];
        if (t * 2 + 1 < n) {
            ds_partialSum[t] += d_vec[t * 2 + 1];
        }
    }

    // Parallel sum reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            ds_partialSum[t] += ds_partialSum[t + stride];
        }
    }

    // The 1st thread saves the sum from shared memory to global memory
    if (t == 0) {
        d_vec[0] = ds_partialSum[0];
    }
}

__global__ void sumRedKernel6(float *d_vec, float *d_out, unsigned int n) {
    extern __shared__ float ds_partialSum[];
    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < n) {
        ds_partialSum[t] = d_vec[i];
        if (i + blockDim.x < n) {
            ds_partialSum[t] += d_vec[i + blockDim.x];
        }
    }

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            ds_partialSum[t] += ds_partialSum[t + stride];
        }
    }

    if (t == 0) {
        d_out[blockIdx.x] = ds_partialSum[0];
    }
}


float sumRed(float *h_vec, unsigned int n, unsigned int version) {
    unsigned int size = n * sizeof(float);
    float *d_vec, *d_out;
    cudaMalloc((void **)&d_vec, size);
    cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);
   
    unsigned int blockSize, gridSize, sharedMemSize;

    switch (version) {
	case 0:
            gridSize = 1;
            blockSize = n;
            sharedMemSize = n * sizeof(float);
            sumRedKernel0<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n);
            break;
    	case 1:
            gridSize = 1;
            blockSize = 1 << ((int) ceil(log2(n)));
            sharedMemSize = blockSize * sizeof(float);
            sumRedKernel1<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n);
            break;
    	case 2:
            gridSize = 1;
            blockSize = (n + 1) / 2;
            sharedMemSize = blockSize * sizeof(float);
            sumRedKernel2<<<gridSize, blockSize, sharedMemSize>>>(d_vec, n);
            break;
    	
        case 6:
        default:
            cudaMalloc((void **)&d_out, size);
            blockSize = 512;
            gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
            sharedMemSize = blockSize * sizeof(float);

            while (n > 1) {
                sumRedKernel6<<<gridSize, blockSize, sharedMemSize>>>(d_vec, d_out, n);
                n = gridSize;
                gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
                cudaMemcpy(d_vec, d_out, n * sizeof(float), cudaMemcpyDeviceToDevice);
            }

            cudaMemcpy(h_vec, d_vec, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_out);
            break;
    }

    cudaFree(d_vec);
    return h_vec[0];
}

int main(int argc, char *argv[]) {
    printf("Enter the number of elements to be summed up: ");
    unsigned int n;
    int ret;
    ret = scanf("%d", &n);
    if (ret != 1) {
        printf("Invalid input. Exiting.\n");
        return 1;
    }

    float *h_vec = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h_vec[i] = (float)i;

    printf("Enter the kernel case (0, 1, 2, or 6): ");
    int kernel_case;
    ret = scanf("%d", &kernel_case);
    if (ret != 1) {
        printf("Invalid input. Exiting.\n");
        free(h_vec);
        return 1;
    }

    float sum = sumRed(h_vec, n, kernel_case);
    printf("sum = %.1f\n", sum);
    free(h_vec);
    return 0;
}
