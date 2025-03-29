#ifdef _GNU_SOURCE
#define PNG_NO_SETJMP
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <immintrin.h>
/* Struct to pass arguments to each thread */
typedef struct {
    int start_row;
    int end_row;
    int width;
    int height;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int* image;
} thread_data_t;

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

/* Function executed by each thread to compute Mandelbrot for a portion of the image */
void* mandelbrot_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    for (int j = data->start_row; j < data->end_row; ++j) {
        double y0 = j * ((data->upper - data->lower) / data->height) + data->lower;

        for (int i = 0; i < data->width; i += 8) {
            // Handle 8 points in parallel using AVX-512
            __m512d x0 = _mm512_set_pd(
                (i + 7) * ((data->right - data->left) / data->width) + data->left,
                (i + 6) * ((data->right - data->left) / data->width) + data->left,
                (i + 5) * ((data->right - data->left) / data->width) + data->left,
                (i + 4) * ((data->right - data->left) / data->width) + data->left,
                (i + 3) * ((data->right - data->left) / data->width) + data->left,
                (i + 2) * ((data->right - data->left) / data->width) + data->left,
                (i + 1) * ((data->right - data->left) / data->width) + data->left,
                i * ((data->right - data->left) / data->width) + data->left
            );

            __m512d y0_vec = _mm512_set1_pd(y0); // Set the same y0 for all elements in the vector
            __m512d x = _mm512_setzero_pd();
            __m512d y = _mm512_setzero_pd();
            __m512d length_squared = _mm512_setzero_pd();

            __m512i repeats = _mm512_setzero_si512(); // Store iteration counts
            const __m512d two = _mm512_set1_pd(2.0);
            const __m512d four = _mm512_set1_pd(4.0);

            for (int iter = 0; iter < data->iters; ++iter) {
                // Calculate x^2 and y^2
                __m512d x2 = _mm512_mul_pd(x, x);
                __m512d y2 = _mm512_mul_pd(y, y);

                // Calculate length_squared = x^2 + y^2
                length_squared = _mm512_add_pd(x2, y2);

                // Mask for points where length_squared < 4.0
                __mmask8 mask = _mm512_cmp_pd_mask(length_squared, four, _CMP_LT_OQ);

                // If all points have escaped, stop early
                if (mask == 0) {
                    break;
                }

                // Update x and y for points that are still in the set
                __m512d xy = _mm512_mul_pd(x, y);
                y = _mm512_fmadd_pd(two, xy, y0_vec); // y = 2 * x * y + y0
                x = _mm512_add_pd(_mm512_sub_pd(x2, y2), x0); // x = x^2 - y^2 + x0

                // Increment repeats for points that are still within bounds
                repeats = _mm512_mask_add_epi32(repeats, mask, repeats, _mm512_set1_epi32(1));
            }

            // Store the iteration counts back to the image buffer
            alignas(64) int repeats_array[8];
            _mm512_store_epi32(repeats_array, repeats);

            // Copy results into the image buffer
            for (int k = 0; k < 8 && (i + k) < data->width; ++k) {
                data->image[j * data->width + (i + k)] = repeats_array[k];
            }
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    struct timespec start, end, temp;
    double time_used;
    clock_gettime(CLOCK_MONOTONIC, &start);
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num_cpus = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", num_cpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* Create threads */
    int num_threads = num_cpus;  // Number of threads will be the number of available CPUs
    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    int rows_per_thread = height / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t == num_threads - 1) ? height : (t + 1) * rows_per_thread;
        thread_data[t].width = width;
        thread_data[t].height = height;
        thread_data[t].iters = iters;
        thread_data[t].left = left;
        thread_data[t].right = right;
        thread_data[t].lower = lower;
        thread_data[t].upper = upper;
        thread_data[t].image = image;

        /* Create thread */
        pthread_create(&threads[t], NULL, mandelbrot_thread, &thread_data[t]);
    }

    /* Join threads */
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    /* Write the image and clean up */
    write_png(filename, iters, width, height, image);
    free(image);
    clock_gettime(CLOCK_MONOTONIC, &end);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
       temp.tv_sec = end.tv_sec-start.tv_sec-1;
       temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
       temp.tv_sec = end.tv_sec - start.tv_sec;
       temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    // printf("%f second\n", time_used);

    return 0;
}
