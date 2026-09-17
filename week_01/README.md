# Week 01: Matrix Multiplication with Pthreads

## Overview
This project implements single-threaded and multi-threaded matrix multiplication in C using POSIX Threads (pthreads). It includes test suites covering varied matrix dimensions and corner cases, element-wise correctness verification between sequential and parallel implementations, and performance benchmark measurements across 1, 4, 16, 32, 64, and 128 threads.

---

## Build and Run

### Requirements
- GCC compiler supporting C99 standard
- POSIX Threads library (-pthread)
- Math library (-lm)

### Compilation
```bash
gcc -O3 -pthread matmul.c -o matmul -lm
```

### Execution
```bash
./matmul
```

---

## Test Cases and Verification

To verify correctness across diverse matrix dimensions and edge conditions, the following test cases are evaluated:

1. **Corner Case: 1x1 Scalar (1 x 1 x 1)**
   - Evaluates single-element matrix computation and validates thread partitioning logic without buffer or index overflow.
2. **Corner Case: Dot Product (1 x 128 x 1)**
   - Multiplies a row vector by a column vector resulting in a 1 x 1 scalar value.
3. **Corner Case: Outer Product (64 x 1 x 64)**
   - Multiplies a column vector by a row vector expanding into a 64 x 64 matrix.
4. **Asymmetric: Skinny and Tall (5 x 500 x 2)**
   - Tests scenarios where row dimensions are smaller than the active thread count.
5. **Prime / Non-divisible Dimensions (37 x 73 x 53)**
   - Verifies dynamic row range chunking when row numbers cannot be evenly partitioned across threads.

**Verification Method**:
For each test case, the multi-threaded output (matmul_multi) across 4 and 16 threads is validated against sequential single-threaded computation (matmul_single). Results are checked element-wise using an absolute floating-point threshold of 1e-4 (fabsf(C1[i] - C2[i]) < 1e-4f). All test cases pass.

---

## Benchmarking & Speedup Analysis

- **Benchmark Configuration**: Matrix dimensions M = 1024, K = 1024, N = 1024
- **Baseline (Single-threaded Time)**: 0.9323 s

| Threads | Time (s) | Speedup | Verified |
| :--- | :--- | :--- | :--- |
| **1** | 0.8750 | 1.07x | PASSED |
| **4** | 0.2964 | 3.15x | PASSED |
| **16** | 0.2344 | 3.98x | PASSED |
| **32** | 0.2358 | 3.95x | PASSED |
| **64** | 0.2270 | 4.11x | PASSED |
| **128** | 0.2294 | 4.06x | PASSED |

---

## Performance Discussion

1. **Scaling from 1 to 4 Threads**:
   - Increasing the thread count from 1 to 4 provides a 3.15x speedup. Because workload is partitioned row-wise and output elements are written independently without shared memory locks, parallel execution utilizes available hardware cores with minimal overhead.

2. **Plateau from 16 to 128 Threads**:
   - **Hardware Core Capacity**: When the number of threads surpasses the host machine's physical CPU cores, thread scheduling and context switching overhead prevent further scaling.
   - **Memory Bandwidth Saturation**: Naive matrix multiplication reads matrix B across all concurrent threads, saturating shared memory bus bandwidth and cache hierarchies.
   - **Thread Management Overhead**: For 64 and 128 threads, thread initialization and joining (pthread_create / pthread_join) consume additional CPU cycles relative to the actual computational payload.