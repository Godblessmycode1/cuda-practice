#include "image_pipeline.h"

// ============================================================================
// HISTOGRAM EQUALIZATION IMPLEMENTATION
// ============================================================================

__global__ void histogram_computation_kernel(cudaTextureObject_t texObj, int* output, int N) {
    __shared__ int histogram[HISTOGRAM_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Initialize shared memory histogram
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        histogram[i] = 0;
    }
    __syncthreads();
    
    // Compute histogram for this block
    if(row < N && col < N) {
        int pixel_value = tex2D<unsigned char>(texObj, col, row);
        atomicAdd(&histogram[pixel_value], 1);
    }
    __syncthreads();
    
    // Reduce to global histogram
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        if(histogram[i] > 0) {
            atomicAdd(&output[i], histogram[i]);
        }
    }
}

__global__ void probability_density_function_kernel(int* input, float* output, int N) {
    int total_pixels = N * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < HISTOGRAM_SIZE) {
        output[tid] = (float)input[tid] / total_pixels;
    }
}

__global__ void intensity_mapping_kernel(float* cdf, int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < HISTOGRAM_SIZE) {
        output[tid] = (int)(cdf[tid] * 255.0f + 0.5f);
    }
}

__global__ void image_transformation_kernel(cudaTextureObject_t texObj, int* lookup_table, unsigned char* output, int N) {
    __shared__ int shared_lut[HISTOGRAM_SIZE];
    
    // Load lookup table into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        shared_lut[i] = lookup_table[i];
    }
    __syncthreads();
    
    // Transform pixels
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        unsigned char old_pixel = tex2D<unsigned char>(texObj, col, row);
        unsigned char new_pixel = shared_lut[old_pixel];
        output[row * N + col] = new_pixel;
    }
}

void histogram_equalization_pipeline(unsigned char* input, unsigned char* output, int N, cudaStream_t stream) {
    // Allocate device memory for intermediate results
    int* d_histogram;
    float* d_pdf;
    float* h_cdf = (float*)malloc(HISTOGRAM_SIZE * sizeof(float));
    float* d_cdf;
    int* d_lookup_table;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, HISTOGRAM_SIZE * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pdf, HISTOGRAM_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cdf, HISTOGRAM_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_lookup_table, HISTOGRAM_SIZE * sizeof(int)));
    
    // Initialize histogram to zero
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_histogram, 0, HISTOGRAM_SIZE * sizeof(int), stream));
    
    // Create pitched memory for texture object
    unsigned char* d_input_pitched;
    size_t input_pitch;
    CHECK_CUDA_ERROR(cudaMallocPitch(&d_input_pitched, &input_pitch, N * sizeof(unsigned char), N));
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(d_input_pitched, input_pitch, input, N * sizeof(unsigned char),
                                      N * sizeof(unsigned char), N, cudaMemcpyDeviceToDevice, stream));
    
    // Create texture object for input
    cudaTextureObject_t texObj = create_texture_object(d_input_pitched, N, N, input_pitch, false);
    
    // Step 1: Compute histogram
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    histogram_computation_kernel<<<gridSize, blockSize, 0, stream>>>(texObj, d_histogram, N);
    
    // Step 2: Compute PDF
    dim3 pdfBlockSize(256);
    dim3 pdfGridSize((HISTOGRAM_SIZE + pdfBlockSize.x - 1) / pdfBlockSize.x);
    probability_density_function_kernel<<<pdfGridSize, pdfBlockSize, 0, stream>>>(d_histogram, d_pdf, N);
    
    // Step 3: Compute CDF on CPU (more efficient for sequential operation)
    float* h_pdf = (float*)malloc(HISTOGRAM_SIZE * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_pdf, d_pdf, HISTOGRAM_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    h_cdf[0] = h_pdf[0];
    for(int i = 1; i < HISTOGRAM_SIZE; i++) {
        h_cdf[i] = h_pdf[i] + h_cdf[i-1];
    }
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cdf, h_cdf, HISTOGRAM_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // Step 4: Create intensity mapping lookup table
    intensity_mapping_kernel<<<pdfGridSize, pdfBlockSize, 0, stream>>>(d_cdf, d_lookup_table);
    
    // Step 5: Transform image
    image_transformation_kernel<<<gridSize, blockSize, 0, stream>>>(texObj, d_lookup_table, output, N);
    
    // Cleanup
    destroy_texture_object(texObj);
    cudaFree(d_input_pitched);
    cudaFree(d_histogram);
    cudaFree(d_pdf);
    cudaFree(d_cdf);
    cudaFree(d_lookup_table);
    free(h_pdf);
    free(h_cdf);
}

// ============================================================================
// LEGACY IMPLEMENTATION (kept for reference and compatibility)
// ============================================================================

__global__ void histogram_computation_legacy(cudaTextureObject_t texObj, int* output, int N){
    __shared__ int histogram[HISTOGRAM_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    // Initialize shared memory histogram
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        histogram[i] = 0;
    }
    __syncthreads();
    
    // Compute histogram for this block
    if(row < N && col < N) {
        int pixel_value = tex2D<unsigned char>(texObj, col, row);  // Note: col, row order for texture
        atomicAdd(&histogram[pixel_value], 1);
    }
    __syncthreads();
    
    // Reduce to global histogram
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        if(histogram[i] > 0) {
            atomicAdd(&output[i], histogram[i]);
        }
    }
}

__global__ void probability_density_function_legacy(int* input, float* output, int N){
    int total_pixels = N * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < HISTOGRAM_SIZE) {
        output[tid] = (float)input[tid] / total_pixels;
    }
}

__host__ void cumulative_distribution_function(float* input, float* cdf){
    cdf[0]=input[0];
    for(int i=1;i<HISTOGRAM_SIZE;i++){
        cdf[i]=input[i]+cdf[i-1];
    }
}

// CDF should be computed on CPU for efficiency
__global__ void intensity_mapping_legacy(float* cdf, int* output){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < HISTOGRAM_SIZE){
        output[tid] = (int)(cdf[tid] * 255.0f + 0.5f);
    }
}

__global__ void image_transformation_legacy(cudaTextureObject_t texObj, int* lookup_table, unsigned char* output, int N){
    __shared__ int shared_lut[HISTOGRAM_SIZE];
    
    // Load lookup table into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;
    
    for(int i = tid; i < HISTOGRAM_SIZE; i += block_size) {
        shared_lut[i] = lookup_table[i];
    }
    __syncthreads();
    
    // Transform pixels
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N){
        unsigned char old_pixel = tex2D<unsigned char>(texObj, col, row);  // col, row for texture
        unsigned char new_pixel = shared_lut[old_pixel];
        output[row * N + col] = new_pixel;
    }
}
