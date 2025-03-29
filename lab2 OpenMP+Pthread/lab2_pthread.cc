#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#define CEILING_POS(X) ((X-(unsigned long long)(X)) > 0 ? (unsigned long long)(X+1) : (unsigned long long)(X))
#define CEILING_NEG(X) ((X-(unsigned long long)(X)) < 0 ? (unsigned long long)(X-1) : (unsigned long long)(X))
#define CEILING(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )

typedef struct {
    unsigned long long r;
    unsigned long long k;
    unsigned long long start_x;
    unsigned long long end_x;
    unsigned long long partial_pixels;
} ThreadData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned long long total_pixels = 0;

void* calculate_pixels(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    unsigned long long local_pixels = 0;
    unsigned long long r_squared = data->r * data->r;
    unsigned long long sqrtlnumber = CEILING(sqrtl(r_squared - data->start_x * data->start_x));  // 首次計算 sqrtl
    local_pixels += sqrtlnumber;
    for (unsigned long long x = data->start_x+1; x < data->end_x; x++) {
       unsigned long long next_value = r_squared - x  * x ;
            
            while (sqrtlnumber * sqrtlnumber >= next_value) {
                sqrtlnumber--;
            }
           
            sqrtlnumber++;
            local_pixels += sqrtlnumber;
            
    }
    
    pthread_mutex_lock(&mutex);
    total_pixels += local_pixels;
    total_pixels %= data->k;
    pthread_mutex_unlock(&mutex);

    return NULL;
}


int main(int argc, char** argv) {
    struct timespec start, end, temp;
    double time_used;
    // clock_gettime(CLOCK_MONOTONIC, &start);
 
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
    int NUM_THREADS=argc;

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
    unsigned long long chunk_size = r / NUM_THREADS;

   
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Create threads and assign them a portion of the loop
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].r = r;
        thread_data[i].k = k;
        thread_data[i].start_x = i * chunk_size;
        if (i == NUM_THREADS - 1) {
            thread_data[i].end_x = r; // Last thread handles any remainder
        } else {
            thread_data[i].end_x = (i + 1) * chunk_size;
        }
        pthread_create(&threads[i], NULL, calculate_pixels, &thread_data[i]);
    }
    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    total_pixels %= k;
    printf("%llu\n", (4 * total_pixels) % k);
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if ((end.tv_nsec - start.tv_nsec) < 0) {
    //    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    //    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    // } else {
    //    temp.tv_sec = end.tv_sec - start.tv_sec;
    //    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    // printf("%f second\n", time_used);

    // pthread_mutex_destroy(&mutex);

    return 0;
    
}
