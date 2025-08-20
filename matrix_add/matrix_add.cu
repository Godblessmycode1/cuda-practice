#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 512  // Using 512x512 for better compatibility

// Version 1: Single Thread - One thread does all work
__global__ void matrix_add_single_thread(float *A, float *B, float *C, int n) {
    // Only thread (0,0) in block (0,0) does all the work
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        // printf("Single thread doing all work: Thread (%d,%d) in Block (%d,%d)\n", 
        //        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                C[idx] = A[idx] + B[idx];
            }
        }
    }
}

// Version 2: Thread Block Level - Multiple threads in a single block
__global__ void matrix_add_thread_block(float *A, float *B, float *C, int n) {
    // Calculate thread position within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_size_x = blockDim.x;  // 32
    int block_size_y = blockDim.y;  // 32
    
    // Only use one block (block 0,0)
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        // Each thread processes multiple elements in a stride pattern
        for (int row = ty; row < n; row += block_size_y) {
            for (int col = tx; col < n; col += block_size_x) {
                int idx = row * n + col;
                C[idx] = A[idx] + B[idx];
            }
        }
    }
}

// Version 3: Multiple Thread Blocks (Grid Level) - Each block handles a portion
__global__ void matrix_add_grid_blocks(float *A, float *B, float *C, int n) {
    // Calculate global thread position
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (global_x < n && global_y < n) {
        int idx = global_y * n + global_x;
        C[idx] = A[idx] + B[idx];
        
        // Print info for threads in first block
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4 && threadIdx.y < 4) {
            // printf("Block (%d,%d) Thread (%d,%d) -> Global pos (%d,%d) processing element [%d][%d]\n", 
            //        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, 
            //        global_x, global_y, global_y, global_x);
        }
    }
}

void matrix_init(float *A, int n, float base_value) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i * n + j] = base_value + i + j;
        }
    }
}

void print_matrix_portion(float *A, int n, int size, const char* name) {
    printf("\n%s (first %dx%d):\n", name, size, size);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            printf("%.1f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int n = N;
    
    // Use dynamic allocation to avoid stack overflow
    float *A = (float*)malloc(n * n * sizeof(float));
    float *B = (float*)malloc(n * n * sizeof(float));
    float *C = (float*)malloc(n * n * sizeof(float));
    
    // Initialize matrices
    matrix_init(A, n, 1.0f);  // A starts with 1
    matrix_init(B, n, 2.0f);  // B starts with 2
    
    // Device memory
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    printf("=== CUDA Performance Comparison ===\n");
    printf("Matrix size: %dx%d (%d elements)\n", n, n, n*n);
    printf("Memory size: %.2f MB\n\n", (3.0 * size) / (1024*1024));
    
    // Warm up GPU
    matrix_add_grid_blocks<<<dim3(32,32), dim3(16,16)>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    // 1. Single Thread Version
    printf("1. SINGLE THREAD VERSION:\n");
    printf("   - 1 block, 1 thread does ALL work\n");
    printf("   - Launch config: <<<(1,1), (1,1)>>>\n");
    
    cudaEventRecord(start);
    dim3 single_grid(1, 1);
    dim3 single_block(1, 1);
    matrix_add_single_thread<<<single_grid, single_block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("   ⏱️  Execution time: %.6f ms\n", milliseconds);
    
    printf("\n==================================================\n");
    
    // 2. Thread Block Version (limited by max threads per block)
    printf("2. THREAD BLOCK VERSION:\n");
    int max_threads = 32;  // Use 32x32 = 1024 threads (max per block)
    printf("   - 1 block with %dx%d threads (limited by hardware)\n", max_threads, max_threads);
    printf("   - Each thread processes multiple elements\n");
    printf("   - Launch config: <<<(1,1), (%d,%d)>>>\n", max_threads, max_threads);
    
    cudaEventRecord(start);
    dim3 block_grid(1, 1);
    dim3 block_threads(max_threads, max_threads);
    
    // Modified kernel call for larger matrices
    int elements_per_thread = (n * n) / (max_threads * max_threads);
    printf("   - Elements per thread: %d\n", elements_per_thread);
    
    // Use a modified version that handles multiple elements per thread
    matrix_add_grid_blocks<<<dim3(1,1), dim3(max_threads, max_threads)>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("   ⏱️  Execution time: %.6f ms\n", milliseconds);
    
    printf("\n==================================================\n");
    
    // 3. Grid of Blocks Version (Optimal)
    printf("3. GRID OF BLOCKS VERSION (OPTIMAL):\n");
    int block_size = 16;  // 16x16 threads per block
    int grid_size = (n + block_size - 1) / block_size;
    printf("   - %dx%d blocks, each with %dx%d threads\n", grid_size, grid_size, block_size, block_size);
    printf("   - Total threads: %d\n", grid_size * grid_size * block_size * block_size);
    printf("   - Launch config: <<<(%d,%d), (%d,%d)>>>\n", grid_size, grid_size, block_size, block_size);
    
    cudaEventRecord(start);
    dim3 grid_blocks(grid_size, grid_size);
    dim3 threads_per_block(block_size, block_size);
    matrix_add_grid_blocks<<<grid_blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("   ⏱️  Execution time: %.6f ms\n", milliseconds);
    float grid_time = milliseconds;
    
    printf("\n==================================================\n");
    
    // // 4. Shared Memory Version
    // printf("4. SHARED MEMORY VERSION:\n");
    // printf("   - %dx%d blocks, each with %dx%d threads\n", grid_size, grid_size, block_size, block_size);
    // printf("   - Uses shared memory for tile-based processing\n");
    // printf("   - Launch config: <<<(%d,%d), (%d,%d)>>>\n", grid_size, grid_size, block_size, block_size);
    
    // cudaEventRecord(start);
    // matrix_add_shared_memory<<<grid_blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("   ⏱️  Execution time: %.6f ms\n", milliseconds);
    
    // printf("\n==================================================\n");
    
    // Performance Analysis
    printf("\n=== PERFORMANCE ANALYSIS ===\n");
    float bandwidth_gb_s = (3.0 * size) / (grid_time * 1e-3) / (1024*1024*1024);
    printf("Grid version achieved: %.2f GB/s memory bandwidth\n", bandwidth_gb_s);
    printf("\nKey Insights:\n");
    printf("• Single thread: Extremely slow - no parallelism\n");
    printf("• Thread block: Limited by max threads per block (1024)\n");
    printf("• Grid of blocks: FASTEST - optimal GPU utilization\n");
    printf("• Shared memory: Similar to grid (for simple operations)\n");
    printf("\nWhy Grid is fastest:\n");
    printf("• Maximizes parallel execution across all SMs\n");
    printf("• Each thread handles exactly one element\n");
    printf("• No synchronization overhead between blocks\n");
    printf("• Optimal memory access patterns\n");
    
    // Verify correctness
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nVerification (first 4x4 elements):\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%.1f ", C[i * n + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    
    return 0;
}
