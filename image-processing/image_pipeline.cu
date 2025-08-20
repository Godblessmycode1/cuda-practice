#include "image_pipeline.h"

// ============================================================================
// COMPLETE PIPELINE IMPLEMENTATION
// ============================================================================

void run_complete_pipeline_single_stream(ImageData* input, ImageData* output, PipelineConfig config, cudaStream_t stream) {
    int N = input->width;
    
    // Allocate device memory for intermediate results
    float* d_gaussian_output;
    float* d_sobel_output;
    unsigned char* d_gaussian_uchar;
    unsigned char* d_sobel_uchar;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_gaussian_output, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sobel_output, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gaussian_uchar, N * N * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sobel_uchar, N * N * sizeof(unsigned char)));
    
    // Stage 1: Gaussian Filter
    if(config.enable_gaussian) {
        setup_gaussian_kernel(config.gaussian_sigma);
        
        // Create padded input for Gaussian (2 pixels padding for 5x5 kernel)
        int gaussian_padded_N = N + 4;
        float* h_input = (float*)malloc(N * N * sizeof(float));
        float* h_gaussian_padded = (float*)malloc(gaussian_padded_N * gaussian_padded_N * sizeof(float));
        
        // Copy input from device to host
        CHECK_CUDA_ERROR(cudaMemcpy2D(h_input, N * sizeof(float), input->data, input->pitch,
                                      N * sizeof(float), N, cudaMemcpyDeviceToHost));
        
        // Pad the image
        for(int i = 0; i < gaussian_padded_N * gaussian_padded_N; i++) {
            h_gaussian_padded[i] = 0.0f;
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                h_gaussian_padded[(i + 2) * gaussian_padded_N + (j + 2)] = h_input[i * N + j];
            }
        }
        
        // Allocate device memory for padded input
        float* d_gaussian_padded;
        size_t gaussian_padded_pitch;
        CHECK_CUDA_ERROR(cudaMallocPitch(&d_gaussian_padded, &gaussian_padded_pitch, 
                                        gaussian_padded_N * sizeof(float), gaussian_padded_N));
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(d_gaussian_padded, gaussian_padded_pitch, 
                                          h_gaussian_padded, gaussian_padded_N * sizeof(float),
                                          gaussian_padded_N * sizeof(float), gaussian_padded_N, 
                                          cudaMemcpyHostToDevice, stream));
        
        // Create texture object and run Gaussian filter
        cudaTextureObject_t gaussian_texObj = create_texture_object(d_gaussian_padded, gaussian_padded_N, 
                                                                   gaussian_padded_N, gaussian_padded_pitch, true);
        gaussian_filter_texture(gaussian_texObj, d_gaussian_output, N, stream);
        
        // Cleanup
        destroy_texture_object(gaussian_texObj);
        cudaFree(d_gaussian_padded);
        free(h_input);
        free(h_gaussian_padded);
    } else {
        // Copy input directly to gaussian output
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(d_gaussian_output, N * sizeof(float), 
                                          input->data, input->pitch,
                                          N * sizeof(float), N, 
                                          cudaMemcpyDeviceToDevice, stream));
    }
    
    // Stage 2: Sobel Edge Detection
    if(config.enable_sobel) {
        setup_sobel_kernels();
        
        // Create padded input for Sobel (1 pixel padding for 3x3 kernel)
        int sobel_padded_N = N + 2;
        float* h_gaussian_result = (float*)malloc(N * N * sizeof(float));
        float* h_sobel_padded = (float*)malloc(sobel_padded_N * sobel_padded_N * sizeof(float));
        
        // Copy Gaussian result from device to host
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_gaussian_result, d_gaussian_output, N * N * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        // Pad the Gaussian result
        for(int i = 0; i < sobel_padded_N * sobel_padded_N; i++) {
            h_sobel_padded[i] = 0.0f;
        }
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                h_sobel_padded[(i + 1) * sobel_padded_N + (j + 1)] = h_gaussian_result[i * N + j];
            }
        }
        
        // Allocate device memory for padded Sobel input
        float* d_sobel_padded;
        size_t sobel_padded_pitch;
        CHECK_CUDA_ERROR(cudaMallocPitch(&d_sobel_padded, &sobel_padded_pitch,
                                        sobel_padded_N * sizeof(float), sobel_padded_N));
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(d_sobel_padded, sobel_padded_pitch,
                                          h_sobel_padded, sobel_padded_N * sizeof(float),
                                          sobel_padded_N * sizeof(float), sobel_padded_N,
                                          cudaMemcpyHostToDevice, stream));
        
        // Create texture object and run Sobel edge detection
        cudaTextureObject_t sobel_texObj = create_texture_object(d_sobel_padded, sobel_padded_N,
                                                                sobel_padded_N, sobel_padded_pitch, true);
        sobel_edge_detection_texture(sobel_texObj, d_sobel_output, N, stream);
        
        // Cleanup
        destroy_texture_object(sobel_texObj);
        cudaFree(d_sobel_padded);
        free(h_gaussian_result);
        free(h_sobel_padded);
    } else {
        // Copy Gaussian output to Sobel output
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sobel_output, d_gaussian_output, N * N * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
    }
    
    // Stage 3: Histogram Equalization
    if(config.enable_histogram) {
        // Convert Sobel output to unsigned char for histogram processing
        float* h_sobel_result = (float*)malloc(N * N * sizeof(float));
        unsigned char* h_sobel_uchar = (unsigned char*)malloc(N * N * sizeof(unsigned char));
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sobel_result, d_sobel_output, N * N * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        // Convert to unsigned char
        convert_float_to_uchar(h_sobel_result, h_sobel_uchar, N);
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sobel_uchar, h_sobel_uchar, N * N * sizeof(unsigned char),
                                        cudaMemcpyHostToDevice, stream));
        
        // Run histogram equalization
        histogram_equalization_pipeline(d_sobel_uchar, output->data_uchar, N, stream);
        
        // Convert result back to float for final output
        unsigned char* h_final_uchar = (unsigned char*)malloc(N * N * sizeof(unsigned char));
        float* h_final_float = (float*)malloc(N * N * sizeof(float));
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_final_uchar, output->data_uchar, N * N * sizeof(unsigned char),
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        convert_uchar_to_float(h_final_uchar, h_final_float, N);
        
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(output->data, output->pitch, h_final_float, N * sizeof(float),
                                          N * sizeof(float), N, cudaMemcpyHostToDevice, stream));
        
        free(h_sobel_result);
        free(h_sobel_uchar);
        free(h_final_uchar);
        free(h_final_float);
    } else {
        // Copy Sobel output to final output
        CHECK_CUDA_ERROR(cudaMemcpy2DAsync(output->data, output->pitch, d_sobel_output, N * sizeof(float),
                                          N * sizeof(float), N, cudaMemcpyHostToDevice, stream));
    }
    
    // Cleanup intermediate buffers
    cudaFree(d_gaussian_output);
    cudaFree(d_sobel_output);
    cudaFree(d_gaussian_uchar);
    cudaFree(d_sobel_uchar);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

ImageData* create_image_data(int width, int height) {
    ImageData* img = (ImageData*)malloc(sizeof(ImageData));
    img->width = width;
    img->height = height;
    
    // Allocate pitched memory for better texture performance
    CHECK_CUDA_ERROR(cudaMallocPitch(&img->data, &img->pitch, width * sizeof(float), height));
    CHECK_CUDA_ERROR(cudaMalloc(&img->data_uchar, width * height * sizeof(unsigned char)));
    
    img->texObj = 0;
    return img;
}

void destroy_image_data(ImageData* img) {
    if(img) {
        if(img->data) cudaFree(img->data);
        if(img->data_uchar) cudaFree(img->data_uchar);
        if(img->texObj) destroy_texture_object(img->texObj);
        free(img);
    }
}

cudaTextureObject_t create_texture_object(void* data, int width, int height, size_t pitch, bool is_float) {
    cudaTextureObject_t texObj = 0;
    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = data;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    
    if(is_float) {
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    } else {
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
    }
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}

void destroy_texture_object(cudaTextureObject_t texObj) {
    if(texObj) {
        CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    }
}

// ============================================================================
// IMAGE I/O FUNCTIONS
// ============================================================================

void load_test_image(float* image, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float value = 0.0f;
            
            // Create checkerboard pattern
            if((i/20 + j/20) % 2 == 0) {
                value += 100.0f;
            }
            
            // Add circular pattern in center
            float center_x = N / 2.0f;
            float center_y = N / 2.0f;
            float distance = sqrt((i - center_x) * (i - center_x) + (j - center_y) * (j - center_y));
            if(distance < N/4) {
                value += 150.0f * exp(-distance * distance / (2 * (N/8) * (N/8)));
            }
            
            // Add some noise
            value += (rand() % 20 - 10);
            
            // Clamp to [0, 255]
            if(value < 0) value = 0;
            if(value > 255) value = 255;
            
            image[i * N + j] = value;
        }
    }
}

void save_image_float(float* image, int N, const char* filename) {
    FILE* file = fopen(filename, "w");
    if(!file) return;
    
    fprintf(file, "P2\n%d %d\n255\n", N, N);
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            int pixel_value = (int)(image[i * N + j]);
            if(pixel_value < 0) pixel_value = 0;
            if(pixel_value > 255) pixel_value = 255;
            fprintf(file, "%d ", pixel_value);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}

void save_image_uchar(unsigned char* image, int N, const char* filename) {
    FILE* file = fopen(filename, "w");
    if(!file) return;
    
    fprintf(file, "P2\n%d %d\n255\n", N, N);
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(file, "%d ", image[i * N + j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}

void convert_float_to_uchar(float* input, unsigned char* output, int N) {
    for(int i = 0; i < N * N; i++) {
        int val = (int)(input[i] + 0.5f);
        if(val < 0) val = 0;
        if(val > 255) val = 255;
        output[i] = (unsigned char)val;
    }
}

void convert_uchar_to_float(unsigned char* input, float* output, int N) {
    for(int i = 0; i < N * N; i++) {
        output[i] = (float)input[i];
    }
}
