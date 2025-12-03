# Square-Root UKF

MKL-accelerated Square-Root Unscented Kalman Filter with Student-t outlier rejection.

## Features

- **Student-t likelihood** — Robust to measurement outliers (configurable ν)
- **Square-root covariance** — Numerical stability via Cholesky factorization
- **Fast** — 1.9 μs/step on Intel CPUs (530K updates/sec)
- **NIS health monitoring** — Windowed statistics for filter divergence detection
- **Python bindings** — ctypes wrapper with NumPy integration
- **Kelly criterion** — Position sizing module that integrates with UKF output

## Performance

| Implementation | Time/step | Steps/sec | vs FilterPy |
|----------------|-----------|-----------|-------------|
| C (MKL batch)  | 1.88 μs   | 530K      | 34× faster  |
| Python ctypes  | 4.15 μs   | 240K      | 15× faster  |
| FilterPy       | 64.6 μs   | 15K       | baseline    |

*Benchmarked on Intel i9-14900K, nx=3, nz=1, Student-t ν=4*

## Quick Start

### C

```c
#include "student_t_srukf.h"

// Create filter: 3 states, 1 measurement, Student-t ν=4
StudentT_SRUKF* ukf = srukf_create(3, 1, 4.0);

// Configure dynamics: x_{k+1} = F @ x_k
double F[9] = {1,0,0, 1,1,0, 0,0,0.95};
srukf_set_dynamics(ukf, F);

// Configure measurement: z_k = H @ x_k  
double H[3] = {1, 0, 0};
srukf_set_measurement(ukf, H);

// Set noise covariances (sqrt form)
double Sq[9] = {0.1,0,0, 0,0.01,0, 0,0,0.05};
double Sr[1] = {1.0};
srukf_set_process_noise(ukf, Sq);
srukf_set_measurement_noise(ukf, Sr);

// Run filter
double z[1];
for (int i = 0; i < n_measurements; i++) {
    z[0] = measurements[i];
    srukf_step(ukf, z);
    
    const double* x = srukf_get_state(ukf);
    printf("State: %.2f, %.4f, %.2f\n", x[0], x[1], x[2]);
}

srukf_destroy(ukf);
```

### Python

```python
from srukf import StudentTSRUKF, Kelly, create_trend_filter

# Quick start with pre-configured trend filter
ukf = create_trend_filter(nu=4.0)

# Run filter
for z in measurements:
    ukf.step([z])
    print(f"State: {ukf.state}, NIS: {ukf.nis:.2f}")

# Kelly position sizing
kelly = Kelly.from_ukf(ukf, fraction=0.5)
print(f"Position: {kelly.position:.3f}, Sharpe: {kelly.sharpe:.2f}")

# Batch processing (faster)
states, covariances, nis_values = ukf.filter(measurements)
```

## Building

### Prerequisites

**Intel oneAPI Math Kernel Library (MKL)**

Download from [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (free).

After installation, MKL is typically located at:
- Windows: `C:\Program Files (x86)\Intel\oneAPI\mkl\<version>\`
- Linux: `/opt/intel/oneapi/mkl/<version>/`

### CMake Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build static library
cmake --build build --config Release

# Build shared library (for Python)
cmake --build build --target student_t_srukf_shared --config Release

# Run tests
cd build
ctest -C Release
```

### Windows Setup

Intel provides a script that sets all required environment variables:

```bash
# Before running Python scripts, call:
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

**Helper scripts included:**

| Script | Purpose |
|--------|---------|
| `run.bat` | Calls `setvars.bat` then runs Python script |
| `deploy_python.bat` | Copies DLL to python folder |

Usage:
```bash
# From project root
deploy_python.bat              # Copy DLL to python/
cd python
..\run.bat compare_ukf.py      # Run with MKL environment
```

### Linux Setup

```bash
# Source MKL environment
source /opt/intel/oneapi/setvars.sh

# Or add to ~/.bashrc:
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
```

## Project Structure

```
Square-Root-UKF/
├── MKL/
│   ├── student_t_srukf.c      # Main implementation
│   ├── student_t_srukf.h      # Public API
│   ├── student_t_srukf.def    # Windows DLL exports
│   └── kelly.h                # Kelly criterion (header-only)
├── python/
│   ├── srukf.py               # Python bindings
│   ├── compare_ukf.py         # FilterPy comparison
│   └── bench_overhead.py      # Python overhead benchmark
├── test/
│   ├── test_srukf.c           # UKF unit tests (11 tests)
│   ├── test_kelly.c           # Kelly unit tests (14 tests)
│   └── bench_srukf.c          # C performance benchmark
├── CMakeLists.txt
├── run.bat                    # Windows Python launcher
├── deploy_python.bat          # DLL deployment script
└── README.md
```

## Tests

### C Tests

```bash
# Build and run
cmake --build build --target test_srukf test_kelly --config Release
./build/Release/test_srukf
./build/Release/test_kelly
```

**test_srukf** (11 tests):
- Filter creation/destruction
- State propagation
- Measurement update
- Student-t weighting
- NIS computation
- Covariance health checks
- Missing data handling
- Serialization/deserialization

**test_kelly** (14 tests):
- Basic Kelly formula
- Tail-adjusted Kelly (Student-t)
- Bayesian variance from log-volatility
- Weak signal filtering
- Multi-asset Kelly
- Transaction cost adjustment
- Health scaling

### Python Tests

```bash
cd python

# Compare with FilterPy (correctness + speed)
python compare_ukf.py

# Measure Python ctypes overhead
python bench_overhead.py
```

**compare_ukf.py**:
- Correctness test vs FilterPy
- Speed benchmark (100 to 100K steps)
- Numerical stability under outliers
- Student-t vs Gaussian robustness

## API Reference

### Core Functions

```c
// Lifecycle
StudentT_SRUKF* srukf_create(int nx, int nz, double nu);
void srukf_destroy(StudentT_SRUKF* ukf);

// Configuration
void srukf_set_state(StudentT_SRUKF* ukf, const double* x0);
void srukf_set_sqrt_cov(StudentT_SRUKF* ukf, const double* S0);
void srukf_set_dynamics(StudentT_SRUKF* ukf, const double* F);
void srukf_set_measurement(StudentT_SRUKF* ukf, const double* H);
void srukf_set_process_noise(StudentT_SRUKF* ukf, const double* Sq);
void srukf_set_measurement_noise(StudentT_SRUKF* ukf, const double* Sr);

// Filtering
void srukf_predict(StudentT_SRUKF* ukf);
void srukf_update(StudentT_SRUKF* ukf, const double* z);
void srukf_step(StudentT_SRUKF* ukf, const double* z);  // predict + update
void srukf_step_batch(StudentT_SRUKF* ukf, const double* z_all, int n_steps);

// State access
const double* srukf_get_state(const StudentT_SRUKF* ukf);
const double* srukf_get_sqrt_cov(const StudentT_SRUKF* ukf);
double srukf_get_nis(const StudentT_SRUKF* ukf);

// Health monitoring
void srukf_enable_nis_tracking(StudentT_SRUKF* ukf, int window_size, double threshold);
bool srukf_nis_healthy(const StudentT_SRUKF* ukf);
```

### Kelly Criterion

```c
#include "kelly.h"

// From UKF state
KellyResult result;
kelly_from_ukf(ukf->x, ukf->S, nx, vel_idx, vol_idx, nu, fraction, &result);
printf("Position: %.3f\n", result.f_final);

// Simple calculation
double f = kelly_simple(mu, sigma, nu, fraction);

// Multi-asset
kelly_multi_asset(mu, S, n_assets, nu, fraction, positions, workspace);
```

## Why Square-Root?

Standard UKF propagates covariance P directly, which can lose positive-definiteness due to numerical errors. Square-root UKF propagates the Cholesky factor S where P = SSᵀ, guaranteeing positive-definiteness by construction.

## Why Student-t?

Gaussian likelihood assigns near-zero probability to outliers, causing the filter to "chase" them. Student-t likelihood with low ν (4-6) has heavier tails — outliers get down-weighted automatically:

```
weight = (ν + nz) / (ν + NIS)
```

Large NIS (outlier) → small weight → measurement ignored.

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

## Author

[Tugbars](https://github.com/Tugbars)