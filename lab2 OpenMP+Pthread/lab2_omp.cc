#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define CEILING_POS(X) ((X-(unsigned long long)(X)) > 0 ? (unsigned long long)(X+1) : (unsigned long long)(X))
#define CEILING_NEG(X) ((X-(unsigned long long)(X)) < 0 ? (unsigned long long)(X-1) : (unsigned long long)(X))
#define CEILING(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )

int main(int argc, char** argv) {
	struct timespec start, end, temp;
    double time_used;
	// clock_gettime(CLOCK_MONOTONIC, &start);
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	int NUM_THREADS=omp_get_max_threads();
	unsigned long long chunk_size = r / NUM_THREADS;

// #pragma omp for schedule(dynamic,chunk_size) nowait
	#pragma omp parallel reduction(+:pixels) num_threads(NUM_THREADS)
		#pragma omp for schedule(static,chunk_size) nowait
		for (unsigned long long x = 0; x < r; x++) {
			unsigned long long y = CEILING(sqrtl(r*r - x*x));
			pixels += y;
                
            
		}
	
	pixels %= k;

	printf("%llu\n", (4 * pixels) % k);
	clock_gettime(CLOCK_MONOTONIC, &end);
    // if ((end.tv_nsec - start.tv_nsec) < 0) {
    //    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    //    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    // } else {
    //    temp.tv_sec = end.tv_sec - start.tv_sec;
    //    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    // printf("%f second\n", time_used);
}
