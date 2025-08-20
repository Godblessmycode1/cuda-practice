# Matrix Addition in CUDA

This folder contains CUDA implementations demonstrating different approaches to parallel matrix addition, showcasing the evolution from basic to optimized GPU programming techniques.

## Files Overview

- `matradd.cu` - Basic matrix addition implementation
- `matrix_add_hierarchy.cu` - Advanced implementation comparing different parallelization strategies
- `matrix` - Compiled executable from `matradd.cu`
- `matrix_hierarchy` - Compiled executable from `matrix_add_hierarchy.cu`

## Learning Objectives

### 1. Basic CUDA Programming Concepts (`matradd.cu`)

This file demonstrates fundamental CUDA programming concepts:

- **Memory Management**: Host-to-device memory allocation and transfer
- **Kernel Launch**: Basic 2D grid and block configuration
- **Thread Indexing**: Using `blockIdx`, `blockDim`, and `threadIdx` for 2D matrix indexing
- **Error Handling**: Basic CUDA memory operations

**Key Learning Points:**
- How to allocate GPU memory with `cudaMalloc()`
- Data transfer between host and device using `cudaMemcpy()`
- 2D thread indexing: `int idx = i * n + j` for row-major matrix storage
- Grid and block dimension calculation for 2D problems

### 2. CUDA Performance Hierarchy (`matrix_add_hierarchy.cu`)

This advanced implementation provides a comprehensive comparison of different parallelization strategies, demonstrating the CUDA execution hierarchy:

#### Version 1: Single Thread Execution
```cuda
__global__ void matrix_add_single_thread(float *A, float *B, float *C, int n)
```
- **Concept**: Only one thread does all the work
- **Launch Config**: `<<<(1,1), (1,1)>>>`
- **Learning**: Shows the importance of parallelization
- **Performance**: Extremely slow - demonstrates why we need parallel computing

#### Version 2: Thread Block Level Parallelization
```cuda
__global__ void matrix_add_thread_block(float *A, float *B, float *C, int n)
```
- **Concept**: Multiple threads within a single block
- **Launch Config**: `<<<(1,1), (32,32)>>>`
- **Learning**: Limited by maximum threads per block (1024)
- **Performance**: Better than single thread but still constrained

#### Version 3: Grid of Blocks 
```cuda
__global__ void matrix_add_grid_blocks(float *A, float *B, float *C, int n)
```
- **Concept**: Multiple blocks across the entire GPU
- **Launch Config**: `<<<(grid_size, grid_size), (16,16)>>>`
- **Learning**: Optimal GPU utilization across all Streaming Multiprocessors (SMs)
- **Performance**: Fastest approach for this problem

## Key CUDA Concepts Demonstrated

### 1. Memory Hierarchy
- **Global Memory**: Main GPU memory (slow but large)
- **Shared Memory**: Fast on-chip memory shared within a block
- **Memory Coalescing**: Efficient memory access patterns

### 2. Execution Hierarchy
- **Thread**: Basic execution unit
- **Block**: Group of threads that can cooperate
- **Grid**: Collection of blocks across the entire GPU

### 3. Performance Optimization
- **Parallelization Strategy**: How to distribute work across threads
- **Memory Bandwidth**: Achieving optimal data transfer rates
- **GPU Utilization**: Maximizing use of available compute resources

### 4. Synchronization
- `__syncthreads()`: Synchronizing threads within a block
- Block-level cooperation using shared memory

## Performance Insights

The hierarchy implementation provides timing comparisons showing:

1. **Single Thread**: Baseline (slowest) - no parallelism
2. **Thread Block**: Limited scalability due to hardware constraints
3. **Grid of Blocks**: Optimal performance - full GPU utilization

**Why Grid of Blocks is Fastest:**
- Maximizes parallel execution across all SMs
- Each thread handles exactly one element
- No synchronization overhead between blocks
- Optimal memory access patterns

## Compilation and Execution

```bash
# Compile basic version
nvcc -o matrix matradd.cu

# Compile hierarchy version
nvcc -o matrix_hierarchy matrix_add_hierarchy.cu

# Run executables

./matrix_hierarchy
```




