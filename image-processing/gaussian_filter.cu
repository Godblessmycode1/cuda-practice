#include "image_pipeline.h"

// Constant memory for Gaussian kernel
__constant__ float gaussian_kernel[GAUSSIAN_KERNEL_SIZE][GAUSSIAN_KERNEL_SIZE];

// ============================================================================
// GAUSSIAN FILTER IMPLEMENTATION
// ============================================================================

__global__ void gaussian_filter_kernel(cudaTextureObject_t texObj, float* output, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        float result = 0.0f;
        
        // For each position in the 5x5 kernel
        for(int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
            for(int j = 0; j < GAUSSIAN_KERNEL_SIZE; j++) {
                // Calculate input position (accounting for padding)
                // The padded image has the original image starting at offset (2,2)
                // So we need to add the output position + kernel offset
                float input_col = col + j;
                float input_row = row + i;
                
                // Use texture memory to fetch pixel value
                // Texture memory handles boundary conditions automatically
                float pixel_value = tex2D<float>(texObj, input_col, input_row);
                
                // Apply kernel weight
                result += pixel_value * gaussian_kernel[i][j];
            }
        }
        
        output[row * N + col] = result;
    }
}

void setup_gaussian_kernel(float sigma) {
    float h_kernel[GAUSSIAN_KERNEL_SIZE][GAUSSIAN_KERNEL_SIZE];
    float sum = 0.0f;
    int center = GAUSSIAN_KERNEL_SIZE / 2;
    
    // Generate kernel values
    for(int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
        for(int j = 0; j < GAUSSIAN_KERNEL_SIZE; j++) {
            int x = i - center;
            int y = j - center;
            h_kernel[i][j] = exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            sum += h_kernel[i][j];
        }
    }
    
    // Normalize kernel
    for(int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++) {
        for(int j = 0; j < GAUSSIAN_KERNEL_SIZE; j++) {
            h_kernel[i][j] /= sum;
        }
    }
    
    // Copy to constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(gaussian_kernel, h_kernel, sizeof(h_kernel)));
}

void gaussian_filter_texture(cudaTextureObject_t texObj, float* output, int N, cudaStream_t stream) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    gaussian_filter_kernel<<<gridSize, blockSize, 0, stream>>>(texObj, output, N);
}

// ============================================================================
// LEGACY IMPLEMENTATION (kept for reference and compatibility)
// ============================================================================

// Legacy kernel for comparison (kept for reference)
__global__ void gaussian_filter_legacy(float* input, float* output, int N)
{
    // Map the thread into output matrix position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Apply Gaussian filter directly (avoiding shared memory race conditions)
    if(row < N && col < N) {  // Boundary check
        float result = 0.0f;
        
        // For each position in the 5x5 kernel
        for(int i = 0; i < GAUSSIAN_KERNEL_SIZE; i++){
            for(int j = 0; j < GAUSSIAN_KERNEL_SIZE; j++){
                // Calculate input position (accounting for padding)
                int input_row = row + i;
                int input_col = col + j;
                int input_idx = input_row * (N + 4) + input_col;  // Note: padded width is N+4
                
                // Apply kernel weight
                result += input[input_idx] * gaussian_kernel[i][j];
            }
        }
        
        output[row * N + col] = result;
    }
}
