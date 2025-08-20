#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>

#define TILE_SIZE 16
#define EPSILON 1e-5

// Original kernel for comparison
__global__ void matrix_mul_original(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    float result = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < N && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            result += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

// Optimization 1: Loop Unrolling
__global__ void matrix_mul_unrolled(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    float result = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (a_row < N && a_col < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Unroll the inner loop (assuming TILE_SIZE = 16)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            result += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}


// CPU matrix multiplication for reference
void cpu_matrix_mul(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Timing function
float time_kernel(void (*kernel)(float*, float*, float*, int), 
                  float* d_A, float* d_B, float* d_C, int N, 
                  dim3 gridSize, dim3 blockSize, const char* name) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Time the kernel
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("%s: %.3f ms (average of 10 runs)\n", name, milliseconds / 10.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 10.0f;
}

// cuBLAS comparison
float time_cublas(cublasHandle_t handle, float* d_A, float* d_B, float* d_C, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    // Time cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                    &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("cuBLAS SGEMM: %.3f ms (average of 10 runs)\n", milliseconds / 10.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / 10.0f;
}

bool test_accuracy(float* gpu_result, float* cpu_result, int N) {
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < N * N; i++) {
        float error = fabs(gpu_result[i] - cpu_result[i]);
        if (error > EPSILON) {
            errors++;
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    
    printf("Elements with error > %e: %d, Max error: %e\n", EPSILON, errors, max_error);
    return errors == 0;
}

int main() {
#ifndef MATRIX_SIZE
    int N = 1024; // Default matrix size if not defined at compile time
#else
    int N = MATRIX_SIZE; // Use compile-time defined matrix size
#endif
    size_t size = N * N * sizeof(float);
    
    printf("Matrix Multiplication Optimization Comparison (N = %d)\n", N);
    printf("=========================================================\n");
    
    // Host matrices
    float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (float)(i + j) / N;
            h_B[i * N + j] = (float)(i - j + N) / N;
        }
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Test different optimizations
    printf("\nPerformance Comparison:\n");
    printf("-----------------------\n");
    
    // Original kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    time_kernel(matrix_mul_original, d_A, d_B, d_C, N, gridSize, blockSize, "Original Tiled");
    
    // Unrolled kernel
    time_kernel(matrix_mul_unrolled, d_A, d_B, d_C, N, gridSize, blockSize, "Loop Unrolled");
    
    // cuBLAS comparison
    cublasHandle_t handle;
    cublasCreate(&handle);
    time_cublas(handle, d_A, d_B, d_C, N);
    cublasDestroy(handle);
    
    // Verify correctness of one kernel
    printf("\nAccuracy Test (Original Kernel):\n");
    printf("--------------------------------\n");
    matrix_mul_original<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    
    printf("Computing CPU reference (this may take a while for large matrices)...\n");
    cpu_matrix_mul(h_A, h_B, h_C_cpu, N);
    
    bool is_correct = test_accuracy(h_C_gpu, h_C_cpu, N);
    printf("%s\n", is_correct ? "✓ Results are correct!" : "✗ Results have errors!");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
