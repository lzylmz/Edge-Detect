#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel to apply the Sobel filter
__global__ void sobelFilter(const unsigned char *gray_img, unsigned char *edge_img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -1 * gray_img[(y - 1) * width + (x - 1)] - 2 * gray_img[y * width + (x - 1)] - 1 * gray_img[(y + 1) * width + (x - 1)]
                + gray_img[(y - 1) * width + (x + 1)] + 2 * gray_img[y * width + (x + 1)] + 1 * gray_img[(y + 1) * width + (x + 1)];

        int gy = -1 * gray_img[(y - 1) * width + (x - 1)] - 2 * gray_img[(y - 1) * width + x] - 1 * gray_img[(y - 1) * width + (x + 1)]
                + gray_img[(y + 1) * width + (x - 1)] + 2 * gray_img[(y + 1) * width + x] + 1 * gray_img[(y + 1) * width + (x + 1)];

        int sum = abs(gx) + abs(gy);
        sum = sum > 255 ? 255 : sum;
        edge_img[y * width + x] = sum;
    } else if (x < width && y < height) {
        edge_img[y * width + x] = gray_img[y * width + x];
    }
}

void processImageOpenMP(const unsigned char *gray_img, unsigned char *edge_img, int width, int height, int num_threads) {
    int gx, gy, sum;
    omp_set_num_threads(num_threads);
    double start_time = omp_get_wtime();

    #pragma omp parallel for private(gx, gy, sum) collapse(2) schedule(guided, 100)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            gx = -1 * gray_img[(y - 1) * width + (x - 1)] - 2 * gray_img[y * width + (x - 1)] - 1 * gray_img[(y + 1) * width + (x - 1)]
                + gray_img[(y - 1) * width + (x + 1)] + 2 * gray_img[y * width + (x + 1)] + 1 * gray_img[(y + 1) * width + (x + 1)];

            gy = -1 * gray_img[(y - 1) * width + (x - 1)] - 2 * gray_img[(y - 1) * width + x] - 1 * gray_img[(y - 1) * width + (x + 1)]
                + gray_img[(y + 1) * width + (x - 1)] + 2 * gray_img[(y + 1) * width + x] + 1 * gray_img[(y + 1) * width + (x + 1)];

            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255 : sum;
            edge_img[y * width + x] = sum;
        }
    }

    double end_time = omp_get_wtime();
    printf("OpenMP edge detection time: %f seconds\n", end_time - start_time);
}

void processImageCUDA(const unsigned char *gray_img, unsigned char *edge_img, int width, int height, int num_blocks, int num_threads_per_block) {
    unsigned char *d_gray_img, *d_edge_img;

    cudaMalloc((void**)&d_gray_img, width * height);
    cudaMalloc((void**)&d_edge_img, width * height);

    cudaMemcpy(d_gray_img, gray_img, width * height, cudaMemcpyHostToDevice);

    dim3 blockSize(num_threads_per_block, num_threads_per_block);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_gray_img, d_edge_img, width, height);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA edge detection time: %f milliseconds\n", milliseconds);
    
    cudaMemcpy(edge_img, d_edge_img, width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_gray_img);
    cudaFree(d_edge_img);
}

void saveCroppedImage(const char* filename, const unsigned char* img, int width, int height, int channels, int crop) {
    int new_width = width - 2 * crop;
    int new_height = height - 2 * crop;
    unsigned char* cropped_img = (unsigned char*)malloc(new_width * new_height * channels);

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            for (int c = 0; c < channels; c++) {
                cropped_img[(y * new_width + x) * channels + c] = img[((y + crop) * width + (x + crop)) * channels + c];
            }
        }
    }

    stbi_write_jpg(filename, new_width, new_height, channels, cropped_img, 100);

    free(cropped_img);
}

int main(int argc, char *argv[]) {
    int width, height, channels;

    if (argc < 6) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <num_threads_OMP> <num_blocks_CUDA> <num_threads_per_block_CUDA>\n", argv[0]);
        return 1;
    }

    unsigned char* gray_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (gray_img == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    unsigned char* edge_img = (unsigned char*)malloc(width * height);
    if (edge_img == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(gray_img);
        return 1;
    }

    int num_threads_OMP = atoi(argv[3]);
    int num_blocks_CUDA = atoi(argv[4]);
    int num_threads_per_block_CUDA = atoi(argv[5]);

    // Create the output filename for the OpenMP result
    char omp_output_file[256];
    snprintf(omp_output_file, sizeof(omp_output_file), "%s_omp.jpg", argv[2]);

    // Process the image using OpenMP
    processImageOpenMP(gray_img, edge_img, width, height, num_threads_OMP);
    saveCroppedImage(omp_output_file, edge_img, width, height, 1, 3);

    // Process the image using CUDA
    processImageCUDA(gray_img, edge_img, width, height, num_blocks_CUDA, num_threads_per_block_CUDA);

    // Create the output filename for the CUDA result
    char cuda_output_file[256];
    snprintf(cuda_output_file, sizeof(cuda_output_file), "%s_cuda.jpg", argv[2]);

    saveCroppedImage(cuda_output_file, edge_img, width, height, 1, 3);

    free(edge_img);
    stbi_image_free(gray_img);

    return 0;
}
