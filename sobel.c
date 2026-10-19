#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TILE_SIZE 16

// ---------------- RGB TO GRAYSCALE ----------------

unsigned char* rgb_to_grayscale(unsigned char* img,
                                int width,
                                int height,
                                int channels)
{
    unsigned char* gray =
        (unsigned char*)malloc(width * height);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * channels;

            unsigned char r = img[idx];
            unsigned char g = img[idx + 1];
            unsigned char b = img[idx + 2];

            gray[y * width + x] =
                (unsigned char)(0.299f * r +
                                0.587f * g +
                                0.114f * b);
        }
    }

    return gray;
}

// ---------------- 5x5 GAUSSIAN BLUR ----------------

void gaussian_blur(unsigned char* input,
                   unsigned char* output,
                   int width,
                   int height)
{
    int kernel[5][5] =
    {
        {1, 4, 6, 4, 1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1, 4, 6, 4, 1}
    };

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 2; ty < height - 2; ty += TILE_SIZE)
    {
        for (int tx = 2; tx < width - 2; tx += TILE_SIZE)
        {
            for (int y = ty;
                 y < ty + TILE_SIZE && y < height - 2;
                 y++)
            {
                int x_limit =
                    (tx + TILE_SIZE < width - 2)
                    ? (tx + TILE_SIZE)
                    : (width - 2);

                #pragma omp simd
                for (int x = tx; x < x_limit; x++)
                {
                    int sum = 0;

                    for (int ky = -2; ky <= 2; ky++)
                    {
                        for (int kx = -2; kx <= 2; kx++)
                        {
                            sum +=
                                input[(y + ky) * width + (x + kx)] *
                                kernel[ky + 2][kx + 2];
                        }
                    }

                    output[y * width + x] =
                        (unsigned char)(sum >> 8);
                }
            }
        }
    }
}

// ---------------- ADAPTIVE THRESHOLD ----------------

int compute_threshold(unsigned char* input,
                      int width,
                      int height)
{
    long long sum = 0;
    long long count = 0;

    #pragma omp parallel for reduction(+:sum,count)
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int gx =
                -1 * input[(y - 1) * width + (x - 1)] +
                 1 * input[(y - 1) * width + (x + 1)] +
                -2 * input[(y) * width + (x - 1)] +
                 2 * input[(y) * width + (x + 1)] +
                -1 * input[(y + 1) * width + (x - 1)] +
                 1 * input[(y + 1) * width + (x + 1)];

            int gy =
                -1 * input[(y - 1) * width + (x - 1)] +
                -2 * input[(y - 1) * width + (x)] +
                -1 * input[(y - 1) * width + (x + 1)] +
                 1 * input[(y + 1) * width + (x - 1)] +
                 2 * input[(y + 1) * width + (x)] +
                 1 * input[(y + 1) * width + (x + 1)];

            sum +=
                (int)sqrtf((float)(gx * gx + gy * gy));

            count++;
        }
    }

    return (int)((sum / count) * 1.5f);
}

// ---------------- SOBEL EDGE DETECTION ----------------

void sobel_parallel(unsigned char* input,
                    unsigned char* output,
                    int width,
                    int height,
                    int threshold)
{
    int threshold2 = threshold * threshold;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ty = 1; ty < height - 1; ty += TILE_SIZE)
    {
        for (int tx = 1; tx < width - 1; tx += TILE_SIZE)
        {
            for (int y = ty;
                 y < ty + TILE_SIZE && y < height - 1;
                 y++)
            {
                int x_limit =
                    (tx + TILE_SIZE < width - 1)
                    ? (tx + TILE_SIZE)
                    : (width - 1);

                #pragma omp simd
                for (int x = tx; x < x_limit; x++)
                {
                    int gx =
                        -1 * input[(y - 1) * width + (x - 1)] +
                         1 * input[(y - 1) * width + (x + 1)] +
                        -2 * input[(y) * width + (x - 1)] +
                         2 * input[(y) * width + (x + 1)] +
                        -1 * input[(y + 1) * width + (x - 1)] +
                         1 * input[(y + 1) * width + (x + 1)];

                    int gy =
                        -1 * input[(y - 1) * width + (x - 1)] +
                        -2 * input[(y - 1) * width + (x)] +
                        -1 * input[(y - 1) * width + (x + 1)] +
                         1 * input[(y + 1) * width + (x - 1)] +
                         2 * input[(y + 1) * width + (x)] +
                         1 * input[(y + 1) * width + (x + 1)];

                    int mag2 =
                        (gx * gx) + (gy * gy);

                    if (mag2 < threshold2)
                    {
                        output[y * width + x] = 0;
                    }
                    else
                    {
                        int magnitude =
                            (int)sqrtf((float)mag2);

                        output[y * width + x] =
                            (unsigned char)
                            (magnitude > 255
                             ? 255
                             : magnitude);
                    }
                }
            }
        }
    }

    // Border correction

    #pragma omp parallel for
    for (int x = 0; x < width; x++)
    {
        output[x] = input[x];

        output[(height - 1) * width + x] =
            input[(height - 1) * width + x];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        output[y * width] =
            input[y * width];

        output[y * width + width - 1] =
            input[y * width + width - 1];
    }
}

// ---------------- MAIN ----------------

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Usage:\n");
        printf("%s input_image output_image threads\n",
               argv[0]);

        return 1;
    }

    int threads = atoi(argv[3]);

    omp_set_num_threads(threads);

    int width, height, channels;

    unsigned char* img =
        stbi_load(argv[1],
                  &width,
                  &height,
                  &channels,
                  0);

    if (!img)
    {
        printf("Failed to load image\n");
        return 1;
    }

    printf("Image: %dx%d (%d channels)\n",
           width,
           height,
           channels);

    printf("Threads: %d\n",
           threads);

    // ---------------- TIMING START ----------------

    double total_start =
        omp_get_wtime();

    // ---------------- GRAYSCALE ----------------

    double gray_start =
        omp_get_wtime();

    unsigned char* gray =
        (channels == 1)
        ? img
        : rgb_to_grayscale(img,
                           width,
                           height,
                           channels);

    double gray_end =
        omp_get_wtime();

    // ---------------- MEMORY ----------------

    unsigned char* blurred =
        (unsigned char*)calloc(width * height, 1);

    unsigned char* output =
        (unsigned char*)calloc(width * height, 1);

    // ---------------- GAUSSIAN BLUR ----------------

    double blur_start =
        omp_get_wtime();

    gaussian_blur(gray,
                  blurred,
                  width,
                  height);

    // Gaussian border correction

    #pragma omp parallel for
    for (int x = 0; x < width; x++)
    {
        blurred[x] = gray[x];

        blurred[(height - 1) * width + x] =
            gray[(height - 1) * width + x];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        blurred[y * width] =
            gray[y * width];

        blurred[y * width + width - 1] =
            gray[y * width + width - 1];
    }

    double blur_end =
        omp_get_wtime();

    // ---------------- THRESHOLD ----------------

    double thresh_start =
        omp_get_wtime();

    int threshold =
        compute_threshold(blurred,
                          width,
                          height);

    double thresh_end =
        omp_get_wtime();

    // ---------------- SOBEL ----------------

    double sobel_start =
        omp_get_wtime();

    sobel_parallel(blurred,
                   output,
                   width,
                   height,
                   threshold);

    double sobel_end =
        omp_get_wtime();

    // ---------------- TOTAL END ----------------

    double total_end =
        omp_get_wtime();

    // ---------------- SAVE OUTPUT ----------------

    stbi_write_png(argv[2],
                   width,
                   height,
                   1,
                   output,
                   width);

    // ---------------- PERFORMANCE REPORT ----------------

    printf("\n===== PERFORMANCE =====\n");

    printf("Grayscale Time: %f sec\n",
           gray_end - gray_start);

    printf("Gaussian Blur Time: %f sec\n",
           blur_end - blur_start);

    printf("Threshold Time: %f sec\n",
           thresh_end - thresh_start);

    printf("Sobel Time: %f sec\n",
           sobel_end - sobel_start);

    printf("Total Time: %f sec\n",
           total_end - total_start);

    printf("=======================\n");

    printf("Adaptive Threshold: %d\n",
           threshold);

    printf("Output image saved successfully\n");

    // ---------------- CLEANUP ----------------

    if (channels != 1)
        free(gray);

    free(blurred);

    free(output);

    stbi_image_free(img);

    return 0;
}