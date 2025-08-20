#include "image_pipeline.h"

// Constant memory for Sobel kernels
__constant__ float sobel_x[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE];
__constant__ float sobel_y[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE];

// ============================================================================
// SOBEL EDGE DETECTION IMPLEMENTATION
// ============================================================================

__global__ void sobel_edge_detection_kernel(cudaTextureObject_t texObj, float* output, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) {
        float result_x = 0.0f;
        float result_y = 0.0f;
        
        for(int i = 0; i < SOBEL_KERNEL_SIZE; i++) {
            for(int j = 0; j < SOBEL_KERNEL_SIZE; j++) {
                int input_row = row + i;  
                int input_col = col + j;  
                float pixel_value = tex2D<float>(texObj, input_col, input_row);
                result_x += pixel_value * sobel_x[i][j];
                result_y += pixel_value * sobel_y[i][j];
            }
        }
        
        output[row * N + col] = sqrtf(result_x * result_x + result_y * result_y);
    }
}

void setup_sobel_kernels() {
    float h_sobel_x[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    float h_sobel_y[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(sobel_x, h_sobel_x, sizeof(h_sobel_x)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(sobel_y, h_sobel_y, sizeof(h_sobel_y)));
}

void sobel_edge_detection_texture(cudaTextureObject_t texObj, float* output, int N, cudaStream_t stream) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    sobel_edge_detection_kernel<<<gridSize, blockSize, 0, stream>>>(texObj, output, N);
}

// ============================================================================
// LEGACY IMPLEMENTATION AND TESTING FUNCTIONS (kept for reference and compatibility)
// ============================================================================

// CPU reference implementation for accuracy testing
void sobel_cpu_reference(float* input, float* output, int N) {
    // Sobel X kernel
    float sobel_x_cpu[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    // Sobel Y kernel
    float sobel_y_cpu[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    for(int row = 0; row < N; row++) {
        for(int col = 0; col < N; col++) {
            float result_x = 0.0f;
            float result_y = 0.0f;
            
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    int input_row = row + i;
                    int input_col = col + j;
                    
                    // Boundary check (assuming padded input)
                    if(input_row >= 0 && input_row < N+2 && input_col >= 0 && input_col < N+2) {
                        float pixel_value = input[input_row * (N+2) + input_col];
                        result_x += pixel_value * sobel_x_cpu[i][j];
                        result_y += pixel_value * sobel_y_cpu[i][j];
                    }
                }
            }
            
            output[row * N + col] = sqrtf(result_x * result_x + result_y * result_y);
        }
    }
}

// Function to create a simple test image with known edges
void create_test_image(float* image, int N) {
    // Create a simple test pattern with vertical and horizontal edges
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(j < N/3) {
                image[i * N + j] = 0.0f;  // Black region
            } else if(j < 2*N/3) {
                image[i * N + j] = 0.5f;  // Gray region
            } else {
                image[i * N + j] = 1.0f;  // White region
            }
            
            // Add horizontal edge
            if(i == N/2) {
                image[i * N + j] = 1.0f - image[i * N + j];
            }
        }
    }
}

// Function to add padding to image for convolution
void add_padding(float* input, float* padded_input, int N) {
    int padded_N = N + 2;
    
    // Initialize padded image with zeros
    for(int i = 0; i < padded_N * padded_N; i++) {
        padded_input[i] = 0.0f;
    }
    
    // Copy original image to center of padded image
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            padded_input[(i+1) * padded_N + (j+1)] = input[i * N + j];
        }
    }
}

// Function to compare GPU and CPU results
bool compare_results(float* gpu_result, float* cpu_result, int N, float tolerance = 1e-5) {
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    
    for(int i = 0; i < N * N; i++) {
        float error = fabsf(gpu_result[i] - cpu_result[i]);
        if(error > tolerance) {
            error_count++;
        }
        max_error = fmaxf(max_error, error);
        avg_error += error;
    }
    
    avg_error /= (N * N);
    
    printf("Accuracy Test Results:\n");
    printf("Max error: %f\n", max_error);
    printf("Average error: %f\n", avg_error);
    printf("Pixels with error > %f: %d out of %d (%.2f%%)\n", 
           tolerance, error_count, N*N, (float)error_count/(N*N)*100);
    
    if(max_error < tolerance) {
        printf("✓ PASS: GPU implementation matches CPU reference\n");
        return true;
    } else {
        printf("✗ FAIL: GPU implementation differs from CPU reference\n");
        return false;
    }
}

__global__ void sobel_edge_detection_legacy(cudaTextureObject_t texObj, float* output, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N){
        float result_x = 0.0f;
        float result_y = 0.0f;
        
        for(int i = 0; i < SOBEL_KERNEL_SIZE; i++){
            for(int j = 0; j < SOBEL_KERNEL_SIZE; j++){
                int input_row = row + i;  
                int input_col = col + j;  
                float pixel_value = tex2D<float>(texObj, input_col, input_row);
                result_x += pixel_value * sobel_x[i][j];
                result_y += pixel_value * sobel_y[i][j];
            }
        }
        
        output[row * N + col] = sqrtf(result_x * result_x + result_y * result_y);
    }
}

// Main test function to verify Sobel edge detection accuracy
int test_sobel_accuracy(int N = 256) {
    printf("Testing Sobel Edge Detection Accuracy (Image size: %dx%d)\n", N, N);
    printf("================================================\n");
    
    // Initialize Sobel kernels
    float h_sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    float h_sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    // Copy kernels to constant memory
    cudaMemcpyToSymbol(sobel_x, h_sobel_x, sizeof(h_sobel_x));
    cudaMemcpyToSymbol(sobel_y, h_sobel_y, sizeof(h_sobel_y));
    
    // Allocate host memory
    float* h_image = (float*)malloc(N * N * sizeof(float));
    float* h_padded_image = (float*)malloc((N+2) * (N+2) * sizeof(float));
    float* h_gpu_result = (float*)malloc(N * N * sizeof(float));
    float* h_cpu_result = (float*)malloc(N * N * sizeof(float));
    
    // Create test image
    create_test_image(h_image, N);
    add_padding(h_image, h_padded_image, N);
    
    // Allocate device memory
    float* d_padded_image;
    float* d_gpu_result;
    size_t pitch;
    cudaMallocPitch(&d_padded_image, &pitch, (N+2) * sizeof(float), N+2);
    cudaMalloc(&d_gpu_result, N * N * sizeof(float));
    
    // Copy padded image to device using pitched memory copy
    cudaMemcpy2D(d_padded_image, pitch, h_padded_image, (N+2) * sizeof(float), 
                 (N+2) * sizeof(float), N+2, cudaMemcpyHostToDevice);
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_padded_image;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = N+2;
    resDesc.res.pitch2D.height = N+2;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    
    // Launch GPU kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    printf("Launching GPU kernel...\n");
    sobel_edge_detection_legacy<<<gridSize, blockSize>>>(texObj, d_gpu_result, N);
    cudaDeviceSynchronize();
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy GPU result back to host
    cudaMemcpy(h_gpu_result, d_gpu_result, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run CPU reference implementation
    printf("Running CPU reference implementation...\n");
    sobel_cpu_reference(h_padded_image, h_cpu_result, N);
    
    // Compare results
    printf("\nComparing GPU and CPU results...\n");
    bool passed = compare_results(h_gpu_result, h_cpu_result, N, 1e-5);
    
    // Print some sample values for debugging - show edge regions
    printf("\nSample values (edge region around N/3 and N/2):\n");
    printf("GPU Result (around vertical edge at col N/3):\n");
    int edge_col = N/3;
    for(int i = N/2-2; i < N/2+3; i++) {
        for(int j = edge_col-2; j < edge_col+3; j++) {
            if(i >= 0 && i < N && j >= 0 && j < N) {
                printf("%.4f ", h_gpu_result[i * N + j]);
            } else {
                printf("------ ");
            }
        }
        printf("\n");
    }
    
    printf("\nCPU Result (around vertical edge at col N/3):\n");
    for(int i = N/2-2; i < N/2+3; i++) {
        for(int j = edge_col-2; j < edge_col+3; j++) {
            if(i >= 0 && i < N && j >= 0 && j < N) {
                printf("%.4f ", h_cpu_result[i * N + j]);
            } else {
                printf("------ ");
            }
        }
        printf("\n");
    }
    
    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFree(d_padded_image);
    cudaFree(d_gpu_result);
    free(h_image);
    free(h_padded_image);
    free(h_gpu_result);
    free(h_cpu_result);
    
    printf("\n================================================\n");
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    
    return passed ? 0 : -1;
}

// Main function for testing (can be used independently)
int main_sobel_test() {
    printf("Sobel Edge Detection Accuracy Test\n");
    printf("==================================\n\n");
    
    // Test with different image sizes
    int test_sizes[] = {64, 128, 256, 512};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int passed_tests = 0;
    
    for(int i = 0; i < num_tests; i++) {
        printf("Test %d/%d: Image size %dx%d\n", i+1, num_tests, test_sizes[i], test_sizes[i]);
        if(test_sobel_accuracy(test_sizes[i]) == 0) {
            passed_tests++;
        }
        printf("\n");
    }
    
    printf("Overall Results: %d/%d tests passed\n", passed_tests, num_tests);
    
    return (passed_tests == num_tests) ? 0 : -1;
}
