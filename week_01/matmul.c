#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    int thread_id;
    int num_threads;
    int M, K, N;
    float *A;
    float *B;
    float *C;
} ThreadArg;

void matmul_single(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void *matmul_worker(void *arg) {
    ThreadArg *data = (ThreadArg *)arg;
    int M = data->M, K = data->K, N = data->N;

    int rows_per_thread = (M + data->num_threads - 1) / data->num_threads;
    int start_row = data->thread_id * rows_per_thread;
    
    if (start_row >= M) {
        return NULL;
    }

    int end_row = start_row + rows_per_thread;
    if (end_row > M) end_row = M;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += data->A[i * K + k] * data->B[k * N + j];
            }
            data->C[i * N + j] = sum;
        }
    }
    return NULL;
}

void matmul_multi(float *A, float *B, float *C, int M, int K, int N, int num_threads) {
    pthread_t threads[num_threads];
    ThreadArg args[num_threads];

    for (int t = 0; t < num_threads; t++) {
        args[t].thread_id = t;
        args[t].num_threads = num_threads;
        args[t].M = M;
        args[t].K = K;
        args[t].N = N;
        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

bool verify_results(float *C1, float *C2, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(C1[i] - C2[i]) > 1e-4f) {
            return false;
        }
    }
    return true;
}

void run_dimension_test(int M, int K, int N, const char *test_name) {
    printf("=== Test: %s (%d x %d x %d) ===\n", test_name, M, K, N);
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_single = (float *)malloc(M * N * sizeof(float));
    float *C_multi = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 10) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 10) * 0.1f;

    matmul_single(A, B, C_single, M, K, N);

    int test_thread_counts[] = {4, 16};
    for (int i = 0; i < 2; i++) {
        int threads = test_thread_counts[i];
        matmul_multi(A, B, C_multi, M, K, N, threads);
        if (verify_results(C_single, C_multi, M * N)) {
            printf("  [%d Threads] Verification PASSED\n", threads);
        } else {
            printf("  [%d Threads] Verification FAILED!\n", threads);
        }
    }

    free(A);
    free(B);
    free(C_single);
    free(C_multi);
}

int main() {
    srand(42);

    printf("---------------- Running Corner Cases ----------------\n");
    run_dimension_test(1, 1, 1, "Corner Case: 1x1 Scalar");
    run_dimension_test(1, 128, 1, "Corner Case: Dot Product (1xK * Kx1)");
    run_dimension_test(64, 1, 64, "Corner Case: Outer Product (Mx1 * 1xN)");
    run_dimension_test(5, 500, 2, "Asymmetric: Skinny and Tall");
    run_dimension_test(37, 73, 53, "Prime / Non-divisible Dimensions");

    printf("\n---------------- Benchmarking & Speedup ----------------\n");
    int M = 1024, K = 1024, N = 1024;
    printf("Benchmark Matrix Dimensions: %dx%dx%d\n", M, K, N);

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_single = (float *)malloc(M * N * sizeof(float));
    float *C_multi = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_single(A, B, C_single, M, K, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double t_single = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Single-threaded Time: %.4f s\n\n", t_single);

    int test_threads[] = {1, 4, 16, 32, 64, 128};
    printf("%-10s | %-12s | %-10s | %-10s\n", "Threads", "Time (s)", "Speedup", "Verified");
    printf("----------------------------------------------------\n");

    for (int idx = 0; idx < 6; idx++) {
        int t = test_threads[idx];
        clock_gettime(CLOCK_MONOTONIC, &start);
        matmul_multi(A, B, C_multi, M, K, N, t);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double speedup = t_single / elapsed;
        bool verified = verify_results(C_single, C_multi, M * N);

        printf("%-10d | %-12.4f | %-9.2fx | %-10s\n", 
               t, elapsed, speedup, verified ? "PASSED" : "FAILED");
    }

    free(A);
    free(B);
    free(C_single);
    free(C_multi);
    return 0;
}
