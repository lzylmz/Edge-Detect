#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1


void seq_edgeDetection(uint8_t* input_image, int width, int height);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int width, height, bpp;

    // Reading the image in grey colors
    uint8_t* input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);

    printf("Width: %d  Height: %d \n", width, height);
    printf("Input: %s , Output: %s  \n", argv[1], argv[2]);

    // start the timer
    double time1 = MPI_Wtime();

    // Applying Sobel Operator
    seq_edgeDetection(input_image, width, height);

    double time2 = MPI_Wtime();
    printf("Elapsed time: %lf \n", time2 - time1);

    // Storing the image
    stbi_write_jpg(argv[2], width, height, CHANNEL_NUM, input_image, 100);
    stbi_image_free(input_image);


    MPI_Finalize();
    return 0;
}


void seq_edgeDetection(uint8_t* input_image, int width, int height) {
    int sobel_x[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};

    int sobel_y[3][3] = {{-1, -2, -1},
                         {0, 0, 0},
                         {1, 2, 1}};

    // Temporary image buffer to store the result of sobel operations
    uint8_t* temp_image = (uint8_t*)malloc(sizeof(uint8_t) * width * height);

    // Applying Sobel operator to each pixel in the image
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sum_x = 0, sum_y = 0;

            // Applying convolution for x and y directions
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum_x += input_image[(i + k) * width + (j + l)] * sobel_x[k + 1][l + 1];
                    sum_y += input_image[(i + k) * width + (j + l)] * sobel_y[k + 1][l + 1];
                }
            }

            // Calculating magnitude of gradient
            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));

            // Clipping the magnitude to ensure it fits in 8 bits (0-255)
            magnitude = magnitude > 255 ? 255 : (magnitude < 0 ? 0 : magnitude);

            // Storing the magnitude in temporary image buffer
            temp_image[i * width + j] = (uint8_t)magnitude;
        }
    }

    // Copying the result from temporary image buffer back to original image buffer
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            input_image[i * width + j] = temp_image[i * width + j];
        }
    }

    // Freeing the temporary image buffer
    free(temp_image);
}