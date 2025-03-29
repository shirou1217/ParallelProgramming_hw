#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <immintrin.h>
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument parsing */
    if (argc != 9) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <output.png> <iters> <left> <right> <lower> <upper> <width> <height>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    // Distribute work among processes
    int rows_per_proc = height / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? height : start_row + rows_per_proc;
    int local_height = end_row - start_row;

    // Allocate memory for local computation
    int* local_image = (int*)malloc(width * local_height * sizeof(int));
    assert(local_image);

    // Measure execution time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    int NUM_THREADS=omp_get_max_threads();
    /* mandelbrot set */
    // #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    // #pragma omp for schedule(static) nowait
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
   for (int j = start_row; j < end_row; ++j) {
        double y0_base = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; i += 8) {
            // Handle 8 points in parallel using AVX-512
            __m512d x0 = _mm512_set_pd(
                (i + 7) * ((right - left) / width) + left,
                (i + 6) * ((right - left) / width) + left,
                (i + 5) * ((right - left) / width) + left,
                (i + 4) * ((right - left) / width) + left,
                (i + 3) * ((right - left) / width) + left,
                (i + 2) * ((right - left) / width) + left,
                (i + 1) * ((right - left) / width) + left,
                i * ((right - left) / width) + left
            );

            __m512d y0 = _mm512_set1_pd(y0_base); // Set the same y0 for all elements in the vector

            // Initialize x and y to 0 for all elements
            __m512d x = _mm512_setzero_pd();
            __m512d y = _mm512_setzero_pd();
            __m512d length_squared = _mm512_setzero_pd();

            // Iteration count for each point
            __m512i repeats = _mm512_setzero_si512();
            const __m512d two = _mm512_set1_pd(2.0);
            const __m512d four = _mm512_set1_pd(4.0);

            for (int iter = 0; iter < iters; ++iter) {
                // Calculate x^2 and y^2
                __m512d x2 = _mm512_mul_pd(x, x);
                __m512d y2 = _mm512_mul_pd(y, y);

                // Calculate length_squared = x^2 + y^2
                length_squared = _mm512_add_pd(x2, y2);

                // Mask for points where length_squared < 4.0
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OQ);

                // If all points are out of bounds, exit early
                if (mask == 0) {
                    break;
                }

                // Update x and y for points that are still in the set
                __m512d xy = _mm512_mul_pd(x, y);
                y = _mm512_fmadd_pd(two, xy, y0); // y = 2 * x * y + y0
                x = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0); // x = x^2 - y^2 + x0

                // Increment repeats for points that are still within bounds
                repeats = _mm512_mask_add_epi32(repeats, mask, repeats, _mm512_set1_epi32(1));
            }

            // Store the iteration counts back to the image buffer
            alignas(64) int repeats_array[8];
            _mm512_store_epi32(repeats_array, repeats);

            for (int k = 0; k < 8 && (i + k) < width; ++k) {
                local_image[(j - start_row) * width + (i + k)] = repeats_array[k];
            }
        }
    }


    // std::ofstream outfile;
    // std::string local_filename = "output_rank_" + std::to_string(rank) + ".txt";
    // outfile.open(local_filename); 
    // for (int j = 0; j < local_height; ++j) {
    //     for (int i = 0; i < width; ++i) {
    //         outfile << local_image[j * width + i] <<" ";
    //     }
    // }
    // outfile << std::endl;
    // outfile.close();

    // Gather results at root process
    int* recvcounts = NULL;
    int* displs = NULL;
    int* local_heights = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        local_heights = (int*)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_height, 1, MPI_INT, local_heights, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = local_heights[i] * width;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    // Allocate memory for the full image at the root process
    int* image = NULL;
    if (rank == 0) {
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
    }

    // Gather the local images to the root process
    MPI_Gatherv(local_image, local_height * width, MPI_INT,
                image, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Root process writes the image and reports execution time
    if (rank == 0) {
        write_png(filename, iters, width, height, image);

         // Open the text file for writing
        // std::ofstream outfile;
        // std::string filename ="hw2b.txt";
        // outfile.open(filename);
        // outfile << "filename" << filename << std::endl;
        // outfile << "iters:" << iters << std::endl;
        // outfile << "width:" << width << std::endl;
        // outfile << "height:" << height << std::endl;
        // outfile << "image:";

        // for (int i = 0; i < width * height; ++i) {
        //     outfile << image[i] << " ";
        //     if ((i + 1) % width == 0) {
        //         outfile << std::endl;  // New line after each row
        //     }
        // }
        // // Close the file
        // outfile.close();


        free(image);
        free(recvcounts);
        free(displs);
        free(local_heights);
        // printf("Total execution time: %f seconds\n", end_time - start_time);
    }

    free(local_image);

    MPI_Finalize();
    return 0;
}
