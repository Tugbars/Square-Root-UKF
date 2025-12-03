/**
 * @file bench_srukf.c
 * @brief Performance benchmark for Student-t SQR UKF
 */

#include "student_t_srukf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#elif defined(__linux__)
    #include <sys/time.h>
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * High-resolution timer
 *───────────────────────────────────────────────────────────────────────────*/

static inline double get_time_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart * 1e9;
#elif defined(__linux__)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#else
    return (double)clock() / CLOCKS_PER_SEC * 1e9;
#endif
}

/*─────────────────────────────────────────────────────────────────────────────
 * Benchmark: Predict step
 *───────────────────────────────────────────────────────────────────────────*/

void bench_predict(int nx, int nz, int iterations) {
    StudentT_SRUKF* ukf = srukf_create(nx, nz, 4.0);
    if (!ukf) {
        fprintf(stderr, "Failed to create UKF\n");
        return;
    }
    
    /* Initialize */
    double* x0 = calloc(nx, sizeof(double));
    double* S0 = calloc(nx * nx, sizeof(double));
    double* F = calloc(nx * nx, sizeof(double));
    double* Sq = calloc(nx * nx, sizeof(double));
    
    for (int i = 0; i < nx; i++) {
        S0[i + i * nx] = 0.1;
        F[i + i * nx] = 1.0;
        Sq[i + i * nx] = 0.01;
    }
    
    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    srukf_set_dynamics(ukf, F);
    srukf_set_process_noise(ukf, Sq);
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        srukf_predict(ukf);
    }
    
    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        srukf_predict(ukf);
    }
    double end = get_time_ns();
    
    double avg_ns = (end - start) / iterations;
    printf("  Predict (nx=%d): %.2f ns/call (%.2f µs)\n", nx, avg_ns, avg_ns / 1000.0);
    
    free(x0);
    free(S0);
    free(F);
    free(Sq);
    srukf_destroy(ukf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * Benchmark: Update step
 *───────────────────────────────────────────────────────────────────────────*/

void bench_update(int nx, int nz, int iterations) {
    StudentT_SRUKF* ukf = srukf_create(nx, nz, 4.0);
    if (!ukf) {
        fprintf(stderr, "Failed to create UKF\n");
        return;
    }
    
    /* Initialize */
    double* x0 = calloc(nx, sizeof(double));
    double* S0 = calloc(nx * nx, sizeof(double));
    double* F = calloc(nx * nx, sizeof(double));
    double* H = calloc(nz * nx, sizeof(double));
    double* Sq = calloc(nx * nx, sizeof(double));
    double* R0 = calloc(nz * nz, sizeof(double));
    double* z = calloc(nz, sizeof(double));
    
    /* Set log-vol state to something reasonable */
    x0[nx - 1] = -3.0;  /* exp(-3) ≈ 0.05 */
    
    for (int i = 0; i < nx; i++) {
        S0[i + i * nx] = 0.1;
        F[i + i * nx] = 1.0;
        Sq[i + i * nx] = 0.01;
    }
    for (int i = 0; i < nz; i++) {
        H[i + i * nx] = 1.0;  /* Observe first nz states */
        R0[i + i * nz] = 0.1;
    }
    
    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    srukf_set_dynamics(ukf, F);
    srukf_set_measurement(ukf, H);
    srukf_set_process_noise(ukf, Sq);
    srukf_set_measurement_noise(ukf, R0);
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        srukf_predict(ukf);
        srukf_update(ukf, z);
    }
    
    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        srukf_update(ukf, z);
    }
    double end = get_time_ns();
    
    double avg_ns = (end - start) / iterations;
    printf("  Update (nx=%d, nz=%d): %.2f ns/call (%.2f µs)\n", nx, nz, avg_ns, avg_ns / 1000.0);
    
    free(x0);
    free(S0);
    free(F);
    free(H);
    free(Sq);
    free(R0);
    free(z);
    srukf_destroy(ukf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * Benchmark: Full step (predict + update)
 *───────────────────────────────────────────────────────────────────────────*/

void bench_step(int nx, int nz, int iterations) {
    StudentT_SRUKF* ukf = srukf_create(nx, nz, 4.0);
    if (!ukf) {
        fprintf(stderr, "Failed to create UKF\n");
        return;
    }
    
    /* Initialize */
    double* x0 = calloc(nx, sizeof(double));
    double* S0 = calloc(nx * nx, sizeof(double));
    double* F = calloc(nx * nx, sizeof(double));
    double* H = calloc(nz * nx, sizeof(double));
    double* Sq = calloc(nx * nx, sizeof(double));
    double* R0 = calloc(nz * nz, sizeof(double));
    double* z = calloc(nz, sizeof(double));
    
    x0[nx - 1] = -3.0;
    
    for (int i = 0; i < nx; i++) {
        S0[i + i * nx] = 0.1;
        F[i + i * nx] = 1.0;
        Sq[i + i * nx] = 0.01;
    }
    for (int i = 0; i < nz; i++) {
        H[i + i * nx] = 1.0;
        R0[i + i * nz] = 0.1;
    }
    
    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    srukf_set_dynamics(ukf, F);
    srukf_set_measurement(ukf, H);
    srukf_set_process_noise(ukf, Sq);
    srukf_set_measurement_noise(ukf, R0);
    
    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        srukf_step(ukf, z);
    }
    
    /* Benchmark */
    double start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        srukf_step(ukf, z);
    }
    double end = get_time_ns();
    
    double avg_ns = (end - start) / iterations;
    double throughput = 1e9 / avg_ns;
    printf("  Step (nx=%d, nz=%d): %.2f ns/call (%.2f µs) | %.2f M steps/sec\n", 
           nx, nz, avg_ns, avg_ns / 1000.0, throughput / 1e6);
    
    free(x0);
    free(S0);
    free(F);
    free(H);
    free(Sq);
    free(R0);
    free(z);
    srukf_destroy(ukf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * Main
 *───────────────────────────────────────────────────────────────────────────*/

int main(void) {
    printf("Student-t SQR UKF Benchmark\n");
    printf("============================\n\n");
    
    int iterations = 100000;
    
    printf("Predict step:\n");
    bench_predict(3, 1, iterations);   /* Typical: trend, velocity, log-vol */
    bench_predict(5, 1, iterations);   /* Larger state */
    bench_predict(10, 1, iterations);  /* Much larger */
    
    printf("\nUpdate step:\n");
    bench_update(3, 1, iterations);
    bench_update(5, 1, iterations);
    bench_update(5, 2, iterations);
    bench_update(10, 3, iterations);
    
    printf("\nFull step (predict + update):\n");
    bench_step(3, 1, iterations);      /* Your typical use case */
    bench_step(5, 1, iterations);
    bench_step(10, 3, iterations);
    
    printf("\n============================\n");
    printf("Benchmark complete.\n");
    
    return 0;
}