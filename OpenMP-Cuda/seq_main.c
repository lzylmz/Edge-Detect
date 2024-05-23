#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    // Load image directly in grayscale
    unsigned char* gray_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (gray_img == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    // Allocate memory for edge image
    unsigned char* edge_img = (unsigned char*)malloc(width * height);
    if (edge_img == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        stbi_image_free(gray_img);
        return 1;
    }

    int gx, gy, sum;

    // Measure the time taken to apply the Sobel filter
    clock_t start_time = clock();

    // Apply Sobel operator
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

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time taken to apply Sobel filter: %f seconds\n", time_taken);

    // Save the cropped image
    saveCroppedImage(argv[2], edge_img, width, height, 1, 3);

    free(edge_img);
    stbi_image_free(gray_img);
    return 0;
}
