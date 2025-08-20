#ifndef IMAGE_PIPELINE_H
#define IMAGE_PIPELINE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Constants
#define GAUSSIAN_KERNEL_SIZE 5
#define SOBEL_KERNEL_SIZE 3
#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 16

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Structure to hold image data
typedef struct {
    float* data;
    unsigned char* data_uchar;
    int width;
    int height;
    size_t pitch;
    cudaTextureObject_t texObj;
} ImageData;

// Structure to hold pipeline configuration
typedef struct {
    float gaussian_sigma;
    bool enable_gaussian;
    bool enable_histogram;
    bool enable_sobel;
    int num_streams;
} PipelineConfig;

// Function declarations for Gaussian filter
void setup_gaussian_kernel(float sigma);
void gaussian_filter_texture(cudaTextureObject_t texObj, float* output, int N, cudaStream_t stream);

// Function declarations for histogram equalization
void histogram_equalization_pipeline(unsigned char* input, unsigned char* output, int N, cudaStream_t stream);

// Function declarations for Sobel edge detection
void setup_sobel_kernels();
void sobel_edge_detection_texture(cudaTextureObject_t texObj, float* output, int N, cudaStream_t stream);

// Utility functions
ImageData* create_image_data(int width, int height);
void destroy_image_data(ImageData* img);
cudaTextureObject_t create_texture_object(void* data, int width, int height, size_t pitch, bool is_float);
void destroy_texture_object(cudaTextureObject_t texObj);

// Image I/O functions
void load_test_image(float* image, int N);
void save_image_float(float* image, int N, const char* filename);
void save_image_uchar(unsigned char* image, int N, const char* filename);
void convert_float_to_uchar(float* input, unsigned char* output, int N);
void convert_uchar_to_float(unsigned char* input, float* output, int N);

// Pipeline functions
void run_sequential_pipeline(ImageData* input, ImageData* output, PipelineConfig config);
void run_streamed_pipeline(ImageData* input, ImageData* output, PipelineConfig config);
void run_complete_pipeline_single_stream(ImageData* input, ImageData* output, PipelineConfig config, cudaStream_t stream);

// Performance measurement
typedef struct {
    float gaussian_time;
    float histogram_time;
    float sobel_time;
    float total_time;
} PerformanceMetrics;

PerformanceMetrics measure_pipeline_performance(ImageData* input, ImageData* output, PipelineConfig config);

#endif // IMAGE_PIPELINE_H
