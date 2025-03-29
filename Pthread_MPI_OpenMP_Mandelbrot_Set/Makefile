CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -fPIC 
hw2a: CFLAGS += -pthread -mavx512f -mavx512dq -march=native
hw2a_SIMD: CFLAGS += -pthread -mavx512f -mavx512dq -march=native
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -mavx512f -mavx512dq -march=native
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a hw2b
# TARGETS = hw2seq hw2a hw2a_SIMD hw2b temp_a temp_b test hw2b_SIMD

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
