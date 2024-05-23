#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#define CHANNEL_NUM 1

void par_edgeDetection(uint8_t* input_image, uint8_t* output_image, int width, int height);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int width, height, bpp;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint8_t* input_image;
    double start_time, end_time; 
    if (rank == 0) {
        input_image = stbi_load(argv[1], &width, &height, &bpp, CHANNEL_NUM);

        if (!input_image) {
            printf("Error loading the image.\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = height / size;
    int total_size = width * rows_per_process;

    uint8_t* local_input = (uint8_t*)malloc(sizeof(uint8_t) * total_size);
    uint8_t* local_output = (uint8_t*)malloc(sizeof(uint8_t) * total_size);

    start_time = MPI_Wtime();

int buffer_size = 0;
int total_buffer_size = buffer_size * 2;

if (rank == 0) {
    for (int dest = 1; dest < size; dest++) {
        MPI_Send(input_image + dest * total_size - buffer_size, (total_size + total_buffer_size) * sizeof(uint8_t), MPI_UNSIGNED_CHAR, dest, 0, MPI_COMM_WORLD);
    }
    memcpy(local_input, input_image, total_size * sizeof(uint8_t));
} else {
    MPI_Recv(local_input - buffer_size, (total_size + total_buffer_size) * sizeof(uint8_t), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

 int performance_cores = 8;
    int core_speed;
    if (rank < performance_cores) {
        core_speed = 5100; 
    } else {
        core_speed = 4100; 
    }

    int local_chunk_height = (int)((double)core_speed / 5100 * rows_per_process);

par_edgeDetection(local_input, local_output, width, local_chunk_height);

if (rank == 0) {
    for (int src = 1; src < size; src++) {
        MPI_Recv(input_image + src * total_size - buffer_size, (total_size + total_buffer_size) * sizeof(uint8_t), MPI_UNSIGNED_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    memcpy(input_image, local_output, total_size * sizeof(uint8_t));
} else {
    MPI_Send(local_output, total_size * sizeof(uint8_t), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
}
    end_time = MPI_Wtime();

    printf("Process %d Elapsed time: %lf \n", rank, end_time - start_time);

    if (rank == 0) {
        stbi_write_jpg(argv[2], width, height, CHANNEL_NUM, input_image, 100);
        stbi_image_free(input_image);
        printf("Width: %d  Height: %d \n", width, height);
        printf("Input: %s , Output: %s  \n", argv[1], argv[2]);
    }

    free(local_input);
    free(local_output);
    MPI_Finalize();
    return 0;
}

void par_edgeDetection(uint8_t* input_image, uint8_t* output_image, int width, int height) {

    int sobel_x[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};

    int sobel_y[3][3] = {{-1, -2, -1},
                         {0, 0, 0},
                         {1, 2, 1}};

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum_x = 0, sum_y = 0;

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int row = i + k;
                    int col = j + l;

                    row = row < 0 ? -row : (row >= height ? 2 * height - row - 2 : row);
                    col = col < 0 ? -col : (col >= width ? 2 * width - col - 2 : col);

                    sum_x += input_image[row * width + col] * sobel_x[k + 1][l + 1];
                    sum_y += input_image[row * width + col] * sobel_y[k + 1][l + 1];
                }
            }

            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));

            magnitude = magnitude > 255 ? 255 : (magnitude < 0 ? 0 : magnitude);

            output_image[i * width + j] = (uint8_t)magnitude;
        }
    }
}
