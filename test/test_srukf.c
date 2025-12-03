/**
 * @file test_srukf.c
 * @brief Basic tests for Student-t SQR UKF
 */

#include "student_t_srukf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(cond, msg)                                           \
    do                                                              \
    {                                                               \
        if (!(cond))                                                \
        {                                                           \
            fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
            return 1;                                               \
        }                                                           \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                                         \
    do                                                                                      \
    {                                                                                       \
        if (fabs((a) - (b)) > (tol))                                                        \
        {                                                                                   \
            fprintf(stderr, "FAIL: %s (line %d): %.6e != %.6e\n", msg, __LINE__, (a), (b)); \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Creation and destruction
 *───────────────────────────────────────────────────────────────────────────*/
int test_create_destroy(void)
{
    printf("Test: create/destroy... ");

    StudentT_SRUKF *ukf = srukf_create(3, 1, 4.0);
    ASSERT(ukf != NULL, "srukf_create failed");

    ASSERT(srukf_get_nx(ukf) == 3, "nx mismatch");
    ASSERT(srukf_get_nz(ukf) == 1, "nz mismatch");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: State initialization
 *───────────────────────────────────────────────────────────────────────────*/
int test_set_state(void)
{
    printf("Test: set_state... ");

    StudentT_SRUKF *ukf = srukf_create(3, 1, 4.0);
    ASSERT(ukf != NULL, "srukf_create failed");

    double x0[3] = {1.0, 0.5, -0.2}; /* trend, velocity, log-vol */
    srukf_set_state(ukf, x0);

    const double *x = srukf_get_state(ukf);
    ASSERT_NEAR(x[0], 1.0, 1e-10, "x[0] mismatch");
    ASSERT_NEAR(x[1], 0.5, 1e-10, "x[1] mismatch");
    ASSERT_NEAR(x[2], -0.2, 1e-10, "x[2] mismatch");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Predict step (linear dynamics)
 *───────────────────────────────────────────────────────────────────────────*/
int test_predict(void)
{
    printf("Test: predict... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Initial state: [trend=1, velocity=0.1, log_vol=-1] */
    double x0[3] = {1.0, 0.1, -1.0};
    srukf_set_state(ukf, x0);

    /* Initial sqrt covariance (diagonal) */
    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.05, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    /* Dynamics: simple integration + AR(1) + mean-reversion */
    /* trend_new = trend + dt * velocity */
    /* velocity_new = 0.95 * velocity */
    /* log_vol_new = 0.98 * log_vol + 0.02 * (-1.0) */
    double dt = 1.0;
    double phi_vel = 0.95;
    double phi_vol = 0.98;

    /* Column-major layout for:
     * F = | 1   dt   0   |
     *     | 0   φv   0   |
     *     | 0   0    φσ  |
     */
    double F[9] = {
        1.0, 0.0, 0.0,    /* Column 0 */
        dt, phi_vel, 0.0, /* Column 1 */
        0.0, 0.0, phi_vol /* Column 2 */
    };
    srukf_set_dynamics(ukf, F);

    /* Process noise */
    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.02};
    srukf_set_process_noise(ukf, Sq);

    /* Predict */
    srukf_predict(ukf);

    /* Check predicted state */
    const double *x = srukf_get_state(ukf);

    /* Expected: trend = 1.0 + 1.0*0.1 = 1.1 */
    ASSERT_NEAR(x[0], 1.1, 1e-6, "predicted trend");
    /* Expected: velocity = 0.95 * 0.1 = 0.095 */
    ASSERT_NEAR(x[1], 0.095, 1e-6, "predicted velocity");
    /* Expected: log_vol = 0.98 * (-1.0) = -0.98 */
    ASSERT_NEAR(x[2], -0.98, 1e-6, "predicted log_vol");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Update step with outlier (Student-t should downweight)
 *───────────────────────────────────────────────────────────────────────────*/
int test_update_outlier(void)
{
    printf("Test: update with outlier... ");

    int nx = 3, nz = 1;
    double nu = 4.0; /* Heavy tails */

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* State: trend=0, velocity=0, log_vol=0 (sigma=1) */
    double x0[3] = {0.0, 0.0, 0.0};
    srukf_set_state(ukf, x0);

    double S0[9] = {
        1.0, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    /* Identity dynamics */
    double F[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    srukf_set_dynamics(ukf, F);

    /* Measurement: observe trend only */
    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);

    /* Process and measurement noise */
    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf, Sq);

    double R0[1] = {1.0}; /* Base measurement noise sqrt */
    srukf_set_measurement_noise(ukf, R0);

    /* Predict first */
    srukf_predict(ukf);

    /* Now update with a large outlier */
    double z_outlier[1] = {10.0}; /* Way out there */
    srukf_update(ukf, z_outlier);

    /* Check Student-t weight - should be significantly < 1 */
    double w = srukf_get_student_weight(ukf);
    printf("(w=%.3f) ", w);
    ASSERT(w < 0.5, "Student-t weight should be low for outlier");

    /* State should not have moved much due to downweighting */
    const double *x = srukf_get_state(ukf);
    ASSERT(fabs(x[0]) < 5.0, "State should not jump fully to outlier");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Update step with normal observation (Student-t should not downweight)
 *───────────────────────────────────────────────────────────────────────────*/
int test_update_normal(void)
{
    printf("Test: update with normal observation... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    double x0[3] = {0.0, 0.0, 0.0};
    srukf_set_state(ukf, x0);

    double S0[9] = {
        1.0, 0.0, 0.0,
        0.0, 0.1, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    double F[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    srukf_set_dynamics(ukf, F);

    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);

    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf, Sq);

    double R0[1] = {1.0};
    srukf_set_measurement_noise(ukf, R0);

    srukf_predict(ukf);

    /* Normal observation */
    double z_normal[1] = {0.5};
    srukf_update(ukf, z_normal);

    /* Check Student-t weight - should be close to 1 */
    double w = srukf_get_student_weight(ukf);
    printf("(w=%.3f) ", w);
    ASSERT(w > 0.8, "Student-t weight should be high for normal observation");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Volatility accessor
 *───────────────────────────────────────────────────────────────────────────*/
int test_volatility(void)
{
    printf("Test: volatility accessor... ");

    StudentT_SRUKF *ukf = srukf_create(3, 1, 4.0);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Set log_vol = log(0.5) */
    double x0[3] = {0.0, 0.0, log(0.5)};
    srukf_set_state(ukf, x0);

    double vol = srukf_get_volatility(ukf);
    ASSERT_NEAR(vol, 0.5, 1e-10, "volatility should be 0.5");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Full cycle (predict + update loop)
 *───────────────────────────────────────────────────────────────────────────*/
int test_full_cycle(void)
{
    printf("Test: full cycle... ");

    int nx = 3, nz = 1;
    double nu = 4.0;
    int n_steps = 100;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Initialize */
    double x0[3] = {0.0, 0.0, log(0.02)}; /* sigma = 0.02 (2% volatility) */
    srukf_set_state(ukf, x0);

    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    /* Random-walk trend with velocity
     * Column-major layout for:
     * F = | 1   1    0    |
     *     | 0   0.95 0    |
     *     | 0   0    0.98 |
     */
    double F[9] = {
        1.0, 0.0, 0.0,  /* Column 0 */
        1.0, 0.95, 0.0, /* Column 1 */
        0.0, 0.0, 0.98  /* Column 2 */
    };
    srukf_set_dynamics(ukf, F);

    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);

    double Sq[9] = {
        0.001, 0.0, 0.0,
        0.0, 0.001, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf, Sq);

    /* R0 is base measurement noise sqrt.
     * Actual R = exp(2*xi) * R0^2 = sigma^2 * R0^2
     * With sigma=0.02, R0=1.0: R = 0.0004, sqrt(R) = 0.02
     */
    double R0[1] = {1.0};
    srukf_set_measurement_noise(ukf, R0);

    /* Simulate */
    double true_trend = 0.0;
    double true_velocity = 0.01;

    for (int i = 0; i < n_steps; i++)
    {
        /* True dynamics */
        true_trend += true_velocity;
        true_velocity *= 0.95;

        /* Generate noisy observation */
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.04;
        double z[1] = {true_trend + noise};

        /* Filter step */
        srukf_step(ukf, z);

        /* Check NIS is reasonable (not exploding) */
        double nis = srukf_get_nis(ukf);
        ASSERT(nis < 100.0, "NIS exploded");
    }

    /* Final state should track trend roughly */
    const double *x = srukf_get_state(ukf);
    printf("(trend_est=%.3f, true=%.3f) ", x[0], true_trend);
    ASSERT(fabs(x[0] - true_trend) < 0.5, "trend estimate should be close");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Missing data handling
 *───────────────────────────────────────────────────────────────────────────*/
int test_missing_data(void)
{
    printf("Test: missing data... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Initialize */
    double x0[3] = {0.0, 0.1, log(0.02)};
    srukf_set_state(ukf, x0);

    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    double F[9] = {
        1.0, 0.0, 0.0,
        1.0, 0.95, 0.0,
        0.0, 0.0, 0.98};
    srukf_set_dynamics(ukf, F);

    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);

    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf, Sq);

    double R0[1] = {1.0};
    srukf_set_measurement_noise(ukf, R0);

    /* Get initial state */
    const double *x_before = srukf_get_state(ukf);
    double trend_before = x_before[0];

    /* Predict only (no observation) */
    srukf_predict_only(ukf);

    /* State should have propagated */
    const double *x_after = srukf_get_state(ukf);
    ASSERT(x_after[0] > trend_before, "trend should increase after predict");

    /* NIS and weight should be at default */
    ASSERT(srukf_get_nis(ukf) == 0.0, "NIS should be 0 after predict_only");
    ASSERT(srukf_get_student_weight(ukf) == 1.0, "weight should be 1 after predict_only");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Covariance repair
 *───────────────────────────────────────────────────────────────────────────*/
int test_covariance_repair(void)
{
    printf("Test: covariance repair... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Set a healthy covariance */
    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    /* Check it's healthy */
    ASSERT(srukf_check_covariance(ukf), "initial covariance should be healthy");

    /* Corrupt the covariance */
    double S_bad[9] = {
        1e-12, 0.0, 0.0, /* Too small diagonal */
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S_bad);

    /* Repair it */
    bool repaired = srukf_repair_covariance(ukf, 1e-6);
    ASSERT(repaired, "repair should detect and fix small diagonal");

    /* Check it's now healthy */
    ASSERT(srukf_check_covariance(ukf), "covariance should be healthy after repair");

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: NIS tracking
 *───────────────────────────────────────────────────────────────────────────*/
int test_nis_tracking(void)
{
    printf("Test: NIS tracking... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf = srukf_create(nx, nz, nu);
    ASSERT(ukf != NULL, "srukf_create failed");

    /* Initialize filter */
    double x0[3] = {0.0, 0.0, log(0.02)};
    srukf_set_state(ukf, x0);

    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf, S0);

    double F[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    srukf_set_dynamics(ukf, F);

    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);

    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf, Sq);

    double R0[1] = {1.0};
    srukf_set_measurement_noise(ukf, R0);

    /* Enable NIS tracking */
    srukf_enable_nis_tracking(ukf, 20, 5.0);

    /* Run some steps */
    for (int i = 0; i < 30; i++)
    {
        double z[1] = {0.01 * i + ((double)rand() / RAND_MAX - 0.5) * 0.04};
        srukf_step(ukf, z);
    }

    /* Get NIS stats */
    SRUKF_NIS_Stats stats;
    srukf_get_nis_stats(ukf, &stats);

    printf("(mean=%.2f, fill=%d) ", stats.mean, stats.window_fill);

    ASSERT(stats.window_fill == 20, "window should be full");
    ASSERT(stats.mean >= 0.0, "mean should be non-negative");

    /* Check health */
    bool healthy = srukf_nis_healthy(ukf);
    /* Should be healthy with normal observations */

    srukf_destroy(ukf);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Test: Serialization
 *───────────────────────────────────────────────────────────────────────────*/
int test_serialization(void)
{
    printf("Test: serialization... ");

    int nx = 3, nz = 1;
    double nu = 4.0;

    StudentT_SRUKF *ukf1 = srukf_create(nx, nz, nu);
    ASSERT(ukf1 != NULL, "srukf_create failed");

    /* Initialize and run some steps */
    double x0[3] = {1.5, 0.1, log(0.02)};
    srukf_set_state(ukf1, x0);

    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1};
    srukf_set_sqrt_cov(ukf1, S0);

    double F[9] = {
        1.0, 0.0, 0.0,
        1.0, 0.95, 0.0,
        0.0, 0.0, 0.98};
    srukf_set_dynamics(ukf1, F);

    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf1, H);

    double Sq[9] = {
        0.01, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.01};
    srukf_set_process_noise(ukf1, Sq);

    double R0[1] = {1.0};
    srukf_set_measurement_noise(ukf1, R0);

    srukf_enable_nis_tracking(ukf1, 10, 5.0);

    /* Run some steps */
    for (int i = 0; i < 15; i++)
    {
        double z[1] = {1.5 + 0.01 * i};
        srukf_step(ukf1, z);
    }

    /* Serialize */
    size_t size = srukf_serialize_size(ukf1);
    ASSERT(size > 0, "serialize size should be positive");

    void *buffer = malloc(size);
    ASSERT(buffer != NULL, "malloc failed");

    size_t written = srukf_serialize(ukf1, buffer, size);
    ASSERT(written == size, "serialize should write exact size");

    /* Create second filter and deserialize */
    StudentT_SRUKF *ukf2 = srukf_create(nx, nz, nu);
    ASSERT(ukf2 != NULL, "srukf_create failed for ukf2");

    /* Must set same model matrices (not serialized) */
    srukf_set_dynamics(ukf2, F);
    srukf_set_measurement(ukf2, H);
    srukf_set_process_noise(ukf2, Sq);
    srukf_set_measurement_noise(ukf2, R0);

    bool ok = srukf_deserialize(ukf2, buffer, size);
    ASSERT(ok, "deserialize should succeed");

    /* Compare states */
    const double *x1 = srukf_get_state(ukf1);
    const double *x2 = srukf_get_state(ukf2);

    for (int i = 0; i < nx; i++)
    {
        ASSERT(fabs(x1[i] - x2[i]) < 1e-10, "states should match after deserialize");
    }

    /* Compare NIS */
    ASSERT(fabs(srukf_get_nis(ukf1) - srukf_get_nis(ukf2)) < 1e-10, "NIS should match");

    free(buffer);
    srukf_destroy(ukf1);
    srukf_destroy(ukf2);

    printf("PASS\n");
    return 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Main
 *───────────────────────────────────────────────────────────────────────────*/
int main(void)
{
    printf("Student-t SQR UKF Tests\n");
    printf("========================\n\n");

    int failed = 0;

    failed += test_create_destroy();
    failed += test_set_state();
    failed += test_predict();
    failed += test_update_outlier();
    failed += test_update_normal();
    failed += test_volatility();
    failed += test_full_cycle();
    failed += test_missing_data();
    failed += test_covariance_repair();
    failed += test_nis_tracking();
    failed += test_serialization();

    printf("\n========================\n");
    if (failed == 0)
    {
        printf("All tests PASSED\n");
    }
    else
    {
        printf("%d test(s) FAILED\n", failed);
    }

    return failed;
}