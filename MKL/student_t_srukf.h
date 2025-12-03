/**
 * @file student_t_srukf.h
 * @brief Student-t Square-Root Unscented Kalman Filter with Intel MKL
 *
 * Designed for quantitative trading systems with:
 * - Fat-tail robust state estimation via Student-t weighting
 * - State-dependent measurement noise (volatility tracking)
 * - BLAS 3 optimized operations
 * - Zero allocations in hot path
 * - AVX2-ready memory alignment (64-byte)
 *
 * State model:
 *   x = [trend, trend_velocity, log_volatility]
 *   Dynamics: linear (random-walk + AR(1) + mean-reverting log-vol)
 *   Measurement: linear mean, state-dependent variance via exp(2*xi)
 *
 * Part of stack: SSA → BOCPD → Student-t SQR UKF → Kelly
 *
 * MKL Configuration:
 *   For Intel i9-14900K: include mkl_config_14900k.h and call mkl_14900k_init_full()
 *   For other CPUs: include mkl_config.h and call mkl_config_init()
 *   Or use the provided run_mkl_14900k.bat/.sh scripts.
 */

#ifndef STUDENT_T_SRUKF_H
#define STUDENT_T_SRUKF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Opaque handle */
    typedef struct StudentT_SRUKF StudentT_SRUKF;

    /*─────────────────────────────────────────────────────────────────────────────
     * LIFECYCLE
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Create a new Student-t SQR UKF instance
     * @param nx State dimension (e.g., 3 for trend, velocity, log-vol)
     * @param nz Measurement dimension (typically 1 for returns)
     * @param nu Student-t degrees of freedom (e.g., 4.0 for heavy tails)
     * @return Allocated filter, or NULL on failure
     */
    StudentT_SRUKF *srukf_create(int nx, int nz, double nu);

    /**
     * @brief Destroy filter and free all memory
     */
    void srukf_destroy(StudentT_SRUKF *ukf);

    /*─────────────────────────────────────────────────────────────────────────────
     * CONFIGURATION (call before first predict/update)
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Set initial state vector
     * @param x0 Initial state [nx × 1], column-major
     */
    void srukf_set_state(StudentT_SRUKF *restrict ukf, const double *restrict x0);

    /**
     * @brief Set initial sqrt covariance
     * @param S0 Lower triangular sqrt covariance [nx × nx], column-major
     */
    void srukf_set_sqrt_cov(StudentT_SRUKF *restrict ukf, const double *restrict S0);

    /**
     * @brief Set state transition matrix
     * @param F Dynamics matrix [nx × nx], column-major
     */
    void srukf_set_dynamics(StudentT_SRUKF *restrict ukf, const double *restrict F);

    /**
     * @brief Set measurement matrix
     * @param H Measurement matrix [nz × nx], column-major
     */
    void srukf_set_measurement(StudentT_SRUKF *restrict ukf, const double *restrict H);

    /**
     * @brief Set process noise sqrt covariance
     * @param Sq Process noise sqrt [nx × nx], column-major, lower triangular
     */
    void srukf_set_process_noise(StudentT_SRUKF *restrict ukf, const double *restrict Sq);

    /**
     * @brief Set base measurement noise (will be scaled by exp(2*xi))
     * @param R0 Base measurement noise sqrt [nz × nz], column-major
     */
    void srukf_set_measurement_noise(StudentT_SRUKF *restrict ukf, const double *restrict R0);

    /**
     * @brief Set UKF tuning parameters
     * @param alpha Spread of sigma points (default: 1e-3)
     * @param beta Prior distribution parameter (default: 2.0 for Gaussian)
     * @param kappa Secondary scaling (default: 0.0)
     */
    void srukf_set_ukf_params(StudentT_SRUKF *ukf, double alpha, double beta, double kappa);

    /**
     * @brief Set which state index is log-volatility
     * @param xi_index Index of log(sigma) in state vector (default: nx-1)
     */
    void srukf_set_vol_index(StudentT_SRUKF *ukf, int xi_index);

    /**
     * @brief Set Student-t degrees of freedom
     * @param nu Degrees of freedom (lower = heavier tails, e.g., 4.0)
     */
    void srukf_set_student_nu(StudentT_SRUKF *ukf, double nu);

    /*─────────────────────────────────────────────────────────────────────────────
     * RUNTIME - HOT PATH (zero allocations)
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Predict step: propagate state through dynamics
     *
     * Computes:
     *   x⁻ = F @ x
     *   S⁻ = sqrt(F @ P @ F' + Q)
     */
    void srukf_predict(StudentT_SRUKF *restrict ukf);

    /**
     * @brief Update step: incorporate measurement with Student-t robustness
     *
     * Computes:
     *   innovation = z - H @ x⁻
     *   d² = innovation' @ S_zz^{-1} @ innovation  (Mahalanobis)
     *   w = (nu + nz) / (nu + d²)                  (Student-t weight)
     *   x = x⁻ + w * K @ innovation
     *   S = downdate(S⁻, sqrt(w) * K @ S_zz)
     *
     * @param z Measurement vector [nz × 1]
     */
    void srukf_update(StudentT_SRUKF *restrict ukf, const double *restrict z);

    /**
     * @brief Combined predict + update (convenience)
     */
    void srukf_step(StudentT_SRUKF *restrict ukf, const double *restrict z);

    /*─────────────────────────────────────────────────────────────────────────────
     * ACCESSORS - For downstream (Kelly sizing, kill switch)
     *───────────────────────────────────────────────────────────────────────────*/

    /** @brief Get current state estimate [nx × 1] */
    const double *srukf_get_state(const StudentT_SRUKF *ukf);

    /** @brief Get current sqrt covariance [nx × nx] lower triangular */
    const double *srukf_get_sqrt_cov(const StudentT_SRUKF *ukf);

    /** @brief Get NIS (Normalized Innovation Squared) for kill switch */
    double srukf_get_nis(const StudentT_SRUKF *ukf);

    /** @brief Get last Student-t weight (1.0 = Gaussian, <1.0 = outlier downweighted) */
    double srukf_get_student_weight(const StudentT_SRUKF *ukf);

    /** @brief Get current volatility estimate exp(xi) */
    double srukf_get_volatility(const StudentT_SRUKF *ukf);

    /** @brief Get Mahalanobis distance squared of last innovation */
    double srukf_get_mahalanobis_sq(const StudentT_SRUKF *ukf);

    /** @brief Get state dimension */
    int srukf_get_nx(const StudentT_SRUKF *ukf);

    /** @brief Get measurement dimension */
    int srukf_get_nz(const StudentT_SRUKF *ukf);

    /*─────────────────────────────────────────────────────────────────────────────
     * DIAGNOSTICS
     *───────────────────────────────────────────────────────────────────────────*/

    /** @brief Get predicted state (before update) [nx × 1] */
    const double *srukf_get_predicted_state(const StudentT_SRUKF *ukf);

    /** @brief Get predicted measurement [nz × 1] */
    const double *srukf_get_predicted_measurement(const StudentT_SRUKF *ukf);

    /** @brief Get last innovation [nz × 1] */
    const double *srukf_get_innovation(const StudentT_SRUKF *ukf);

    /** @brief Get last Kalman gain [nx × nz] */
    const double *srukf_get_kalman_gain(const StudentT_SRUKF *ukf);

    /*─────────────────────────────────────────────────────────────────────────────
     * MISSING DATA HANDLING
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Predict-only step (no observation available)
     *
     * Use for: market closures, holidays, missing data, forecasting
     * Equivalent to: srukf_predict() without srukf_update()
     * Covariance grows according to process noise.
     */
    void srukf_predict_only(StudentT_SRUKF *restrict ukf);

    /**
     * @brief Update with partial observation (some components missing)
     *
     * @param z Measurement vector [nz × 1] (NAN for missing)
     * @param mask Boolean mask [nz × 1] (true = observed, false = missing)
     * @return Number of components actually updated
     *
     * Internally reduces measurement dimension for this step.
     * If all masked, equivalent to predict-only.
     */
    int srukf_update_partial(StudentT_SRUKF *restrict ukf,
                             const double *restrict z,
                             const bool *restrict mask);

    /*─────────────────────────────────────────────────────────────────────────────
     * COVARIANCE HEALTH & REPAIR
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Check if sqrt covariance is well-conditioned
     * @return true if healthy, false if needs repair
     */
    bool srukf_check_covariance(const StudentT_SRUKF *ukf);

    /**
     * @brief Repair ill-conditioned covariance
     *
     * Operations:
     *   - Enforces minimum diagonal (prevents collapse)
     *   - Re-triangularizes if needed
     *   - Ensures positive definiteness
     *
     * @param min_diag Minimum allowed diagonal value (e.g., 1e-8)
     * @return true if repair was needed, false if already healthy
     */
    bool srukf_repair_covariance(StudentT_SRUKF *ukf, double min_diag);

    /**
     * @brief Check for state bounds (detect numerical blow-up)
     * @param max_abs Maximum absolute value for any state component
     * @return true if within bounds, false if blown up
     */
    bool srukf_check_state_bounds(const StudentT_SRUKF *ukf, double max_abs);

    /*─────────────────────────────────────────────────────────────────────────────
     * WINDOWED NIS STATISTICS (for kill switch)
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief NIS statistics structure for kill switch integration
     */
    typedef struct
    {
        double mean;       /* Windowed mean of NIS (expected: nz) */
        double variance;   /* Windowed variance of NIS */
        double trend;      /* Recent trend (positive = degrading) */
        double max_recent; /* Maximum NIS in window */
        int n_outliers;    /* Count of NIS > threshold in window */
        int window_fill;   /* How many samples in window (0 to window_size) */
    } SRUKF_NIS_Stats;

    /**
     * @brief Enable windowed NIS tracking
     * @param window_size Rolling window size (e.g., 50-100)
     * @param outlier_threshold NIS threshold for outlier counting (e.g., 3*nz)
     */
    void srukf_enable_nis_tracking(StudentT_SRUKF *ukf, int window_size, double outlier_threshold);

    /**
     * @brief Get current NIS statistics
     * @param stats Output structure (caller-allocated)
     */
    void srukf_get_nis_stats(const StudentT_SRUKF *ukf, SRUKF_NIS_Stats *stats);

    /**
     * @brief Check if NIS indicates model health
     * @return true if healthy, false if kill switch should consider action
     *
     * Checks: mean not too high, not trending up, outlier rate acceptable
     */
    bool srukf_nis_healthy(const StudentT_SRUKF *ukf);

    /*─────────────────────────────────────────────────────────────────────────────
     * STATE MANAGEMENT & SERIALIZATION
     *───────────────────────────────────────────────────────────────────────────*/

    /**
     * @brief Reset filter to initial state without reallocation
     * @param x0 New initial state [nx × 1]
     * @param S0 New initial sqrt covariance [nx × nx]
     *
     * Clears all diagnostics and NIS history.
     */
    void srukf_reset(StudentT_SRUKF *ukf,
                     const double *restrict x0,
                     const double *restrict S0);

    /**
     * @brief Get serialization size in bytes
     * @return Number of bytes needed for srukf_serialize()
     */
    size_t srukf_serialize_size(const StudentT_SRUKF *ukf);

    /**
     * @brief Serialize filter state to buffer
     * @param buffer Output buffer (must be >= srukf_serialize_size())
     * @param buffer_size Size of buffer
     * @return Bytes written, or 0 on failure
     *
     * Saves: state, covariance, diagnostics, NIS history
     * Does NOT save: model matrices (F, H, Q, R) - assumed constant
     */
    size_t srukf_serialize(const StudentT_SRUKF *ukf,
                           void *buffer,
                           size_t buffer_size);

    /**
     * @brief Deserialize filter state from buffer
     * @param buffer Input buffer from srukf_serialize()
     * @param buffer_size Size of buffer
     * @return true on success, false on failure (size mismatch, corruption)
     *
     * Filter dimensions (nx, nz) must match.
     */
    bool srukf_deserialize(StudentT_SRUKF *ukf,
                           const void *buffer,
                           size_t buffer_size);

    /**
     * @brief Get filter version for compatibility checking
     * @return Version number (increment on breaking changes)
     */
    uint32_t srukf_version(void);

#ifdef __cplusplus
}
#endif

#endif /* STUDENT_T_SRUKF_H */