# Parallel Programming Course Projects

This repository contains the projects I completed during my Parallel Programming course. Each project explores different parallel computing techniques and frameworks, demonstrating my skills in developing high-performance software solutions.

## Overview of Projects

| Project | Key Technologies | Description |
|---------|-----------------|-------------|
| [Odd-Even Sort](#hw1-odd-even-sort) | MPI, Parallel Sorting | Implemented the odd-even sort algorithm using MPI |
| [Mandelbrot Set](#hw2-mandelbrot-set) | Pthread, OpenMP, MPI, SIMD | Generated Mandelbrot set images using various parallel techniques |
| [All-Pairs Shortest Path](#hw3-all-pairs-shortest-path) | OpenMP, CUDA, Multi-GPU | Implemented Floyd-Warshall algorithm using various parallelization strategies |
| [Attention Mechanism](#hw4-attention-mechanism) | CUDA, GPU Optimization | Implemented and optimized attention mechanisms with FlashAttention |
| [UCX Programming](#hw5-ucx-programming) | UCX, UCP, Transport Layer | Analyzed UCX architecture and implemented features |
| [FlashAttention Analysis](#lab5-flashattention-analysis) | PyTorch, GPU Acceleration | Compared performance between PyTorch and FlashAttention v2 |

## HW1: Odd-Even Sort

### Technologies Used
- **MPI (Message Passing Interface)** for inter-process communication
- **C++ STL** (Standard Template Library) for local sorting operations
- **Intel APS** (Application Performance Snapshot) for profiling

### Implementation Details
- Distributed odd-even sort across multiple processes
- Handled arbitrary number of input items and processes
- Used optimized communication patterns with `MPI_Sendrecv` instead of separate `MPI_Send` and `MPI_Recv`
- Added early termination using `MPI_Allreduce` to check if sorting is complete
- Optimized by checking if adjacent ranks' data is already sorted

### Performance Analysis
- Identified communication as the main bottleneck
- Analyzed CPU time, communication time, and I/O time
- Experimented with various numbers of processes to evaluate scalability
- Achieved optimization by replacing separate MPI_Send/Recv with MPI_Sendrecv

### Key Learnings
- Practical understanding of MPI programming patterns
- Performance analysis techniques using Intel APS
- Runtime breakdown into CPU, Communication, and I/O time
- Scalability challenges in distributed sorting algorithms

## HW2: Mandelbrot Set

### Technologies Used
- **Pthread** for multi-threaded computation
- **OpenMP** for simplified parallelism
- **MPI** for multi-process computation
- **Hybrid parallel programming** (MPI + OpenMP)
- **AVX-512 SIMD** instructions for vectorized computation
- **Intel VTune** for performance profiling

### Implementation Details
- Three implementations:
  1. **Pthread**: Divided workload based on image height among threads
  2. **Hybrid (MPI + OpenMP)**: Combined process and thread level parallelism
  3. **SIMD Optimization**: Used AVX-512 instructions to process multiple pixels simultaneously

### Performance Analysis
- Conducted comprehensive performance measurements using Intel VTune
- Analyzed scalability with varying numbers of threads and processes
- Identified the mandelbrot set computation as the main performance bottleneck
- Implemented SIMD optimization for significant performance improvement
- Evaluated load balancing across threads/processes

### Key Learnings
- Multi-level parallelism techniques
- Performance profiling with Intel VTune
- SIMD programming using AVX-512 instructions
- Load balancing strategies for embarrassingly parallel problems
- Strong and weak scaling analysis

## HW3: All-Pairs Shortest Path

### Technologies Used
- **OpenMP** for thread-level parallelism (HW3-1)
- **CUDA** for GPU acceleration (HW3-2)
- **Multi-GPU** with CUDA for distributed computation (HW3-3)
- **AMD GPU** environment for cross-platform comparison

### Implementation Details
- **HW3-1 (OpenMP)**: 
  - Used `#pragma omp parallel sections` for Phase 2 optimization
  - Used `#pragma omp parallel for collapse(2) schedule(dynamic)` for Phase 3 and inner calculations
  - Parallelized the outer loop structures of the Floyd-Warshall algorithm

- **HW3-2 (CUDA Single GPU)**: 
  - Implemented block-based Floyd-Warshall algorithm with CUDA
  - Used matrix blocking with BLOCK_SIZE=64 for efficient parallelization
  - Employed shared memory for faster block operations
  - Created specialized kernels for each phase of the algorithm

- **HW3-3 (CUDA Multi-GPU)**:
  - Distributed computation across two GPUs with round-robin scheduling
  - Synchronized data between GPUs using `cudaMemcpyPeer`
  - Alternated computation between GPUs for even/odd rounds

### Performance Analysis
- Profiled with NSight to identify performance bottlenecks
- Analyzed occupancy, SM efficiency, and memory throughput across phases
- Experimented with different blocking factors (B=8, B=16, B=32, B=64)
- Found that B=64 provided optimal shared memory bandwidth utilization
- Discovered that single GPU outperformed dual GPU due to synchronization overhead
- Compared NVIDIA GTX 1080 with AMD MI210 GPU, finding AMD GPUs 2-3x faster (single) and 11x faster (multi)

### Key Learnings
- Block-based algorithm decomposition for parallel processing
- GPU memory hierarchy optimization techniques
- Data padding strategies to align with hardware requirements
- Multi-GPU communication and synchronization challenges
- Cross-platform GPU performance characteristics

## HW4: Attention Mechanism

### Technologies Used
- **CUDA** for GPU-accelerated computation
- **Shared memory** optimization techniques
- **Matrix tiling** for efficient GPU utilization
- **Warp-level reduction** for performance enhancement

### Implementation Details
- **Sequential Attention (seq-attention.c)**:
  - Implemented basic attention mechanism with CUDA kernels
  - Used warp-reduce technique for optimizing reduction operations
  - Created specialized kernels: `QKDotAndScalarKernel`, `SoftMaxKernel`, and `MulAttVKernel`
  - Employed tree-reduction in shared memory for efficient computation

- **FlashAttention (seq-flashattention.c)**:
  - Implemented the FlashAttention algorithm with blocking strategy
  - Used matrix tiling to efficiently utilize GPU memory hierarchy
  - Processed multiple batches simultaneously
  - Implemented specialized kernels for scaling factors (ℓ and m) calculation
  - Used blocking factors B_r=32 and B_c=32 with TILE_SIZE=16 for optimal performance

### Performance Analysis
- Employed profiling to analyze resource utilization and execution efficiency
- Found that shared memory and warp-reduce techniques significantly improved performance
- Identified that SoftMaxKernel and QKDotAndScalarKernel achieved higher GPU utilization
- Observed higher achieved occupancy and SM efficiency in kernels with warp-level reduction
- Measured memory throughput improvements through shared memory usage

### Key Learnings
- Advanced CUDA programming techniques
- Memory hierarchy optimization for GPU computing
- SRAM utilization for performance enhancement
- Matrix tiling and blocking strategies for efficient parallel computation
- Warp-level reduction techniques for optimized GPU computation

## HW5: UCX Programming

### Technologies Used
- **UCX (Unified Communication X)** framework
- **UCP (UCX Protocol)** API
- **Transport Layer Security (TLS)** in networking

### Implementation Details
- Analyzed UCP architecture, including the relationship between UCP Objects:
  - `ucp_context`
  - `ucp_worker`
  - `ucp_ep`
- Modified UCX source code to implement special features for printing TLS information
- Analyzed UCX transport layer implementation details

### Performance Analysis
- Evaluated different TLS implementations for performance
- Tested multi-node performance using OSU benchmarks
- Compared latency and bandwidth metrics for different configurations

### Key Learnings
- Deep understanding of UCX architecture and communication layers
- Experience with tracing code in large library projects
- Network communication protocols and transport layers
- Optimization techniques for high-performance computing (HPC) communication

## Lab5: FlashAttention Analysis

### Technologies Used
- **PyTorch** deep learning framework
- **FlashAttention v2** optimized attention implementation
- **GPU performance analysis**

### Key Findings
- FlashAttention v2 showed significantly better performance than PyTorch's native attention:
  - 9x faster forward propagation
  - 4.5x faster backward propagation
  - 10x higher FLOPS in forward stage, 5x in backward stage
  - 86% reduction in memory usage

### Parameter Analysis
- Analyzed performance across different batch sizes, sequence lengths, number of heads, and embedding dimensions
- FlashAttention maintained better performance scaling with increased parameters

### Key Learnings
- Performance optimization techniques for attention mechanisms
- Memory usage patterns in transformer models
- GPU computation efficiency analysis
- Impact of different parameters on model performance

## Skills Summary

Through these projects, I've developed expertise in:

1. **Parallel Programming Paradigms**:
   - Multi-threading (Pthread)
   - Shared memory parallelism (OpenMP)
   - Message passing (MPI)
   - Hybrid parallelism
   - SIMD vectorization
   - GPU programming (CUDA)

2. **Performance Analysis**:
   - Profiling with Intel VTune, APS, and NVIDIA NSight
   - Identifying performance bottlenecks
   - Optimizing communication patterns
   - Analyzing memory usage and bandwidth
   - GPU kernel optimization

3. **HPC Communication**:
   - Understanding network communication layers
   - Working with high-performance frameworks like UCX
   - Optimizing data transfer between nodes
   - Multi-GPU coordination strategies

4. **GPU Optimization Techniques**:
   - Blocking and tiling for memory hierarchy
   - Shared memory utilization
   - Warp-level reduction
   - Memory coalescing
   - Multiple GPU coordination

5. **Deep Learning Optimization**:
   - Analyzing attention mechanism implementations
   - Understanding GPU computation patterns
   - Evaluating memory efficiency in model implementations
   - Performance scaling with increasing model parameters

These projects demonstrate my ability to develop and optimize parallel software for high-performance computing environments, with a focus on scalability, efficiency, and performance analysis.
