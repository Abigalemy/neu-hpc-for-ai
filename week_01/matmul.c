#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

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
        args[t].M = M; args[t].K = K; args[t].N = N;
        args[t].A = A; args[t].B = B; args[t].C = C;
        pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

int main() {
    int M = 1024, K = 1024, N = 1024;
    printf("Matrix multiplication test: %dx%dx%d\n", M, K, N);

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C_single = (float *)malloc(M * N * sizeof(float));
    float *C_multi = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 2.0f;


    matmul_single(A, B, C_single, M, K, N);


    int test_threads[] = {1, 4, 16, 32, 64, 128};
    for (int idx = 0; idx < 6; idx++) {
        int t = test_threads[idx];
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        matmul_multi(A, B, C_multi, M, K, N, t);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Threads: %3d | Time: %.4f s\n", t, elapsed);
    }

    free(A); free(B); free(C_single); free(C_multi);
    return 0;
}
