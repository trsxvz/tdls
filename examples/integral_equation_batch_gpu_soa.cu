/// \file
/// \brief Example: parameter sweep of Love's integral equation on GPU,
/// with the dense Nystroem systems materialized in device memory in SoA
/// layout and solved with the runtime TiledLUpp solver at batch stride.
/// \author Tristan Chenaille
///
/// Companion of integral_equation_batch_gpu.cu: same sweep, same batch
/// size, opposite placement. The runtime solver has no residency
/// booleans; the placement of the operands is a property of the
/// pointers and strides handed to it. Here the batch of systems is
/// materialized in device memory structure-of-arrays: element k of
/// system t sits at base[k * B + t], so the solver is called with
/// element stride B and consecutive threads touch consecutive
/// addresses, every access coalesced.
///
/// The batch is produced by an assembly kernel and consumed by a solve
/// kernel, the standard pattern when assembly and solve do not want the
/// same launch configuration. The price of materialization is that the
/// batch occupies device memory, which the thread-local variant never
/// does.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tdls/tdls.hpp>

#include "gpu_runtime.hpp"

namespace {

using Solver = tdls::TiledLUppSolverDynamic<double>;

constexpr int instances = 1 << 17; ///< plate separations in the sweep
constexpr int max_n     = 40;      ///< bound of the runtime resolution
constexpr double d_min  = 0.5;     ///< smallest plate separation
constexpr double d_max  = 2.0;     ///< largest plate separation

/// \brief Love kernel of the parallel-plate capacitor.
/// \param[in] x first quadrature point
/// \param[in] y second quadrature point
/// \param[in] d plate separation
/// \return the kernel value
__device__ double love_kernel(const double x, const double y, const double d) {
    return d / ((d * d + (x - y) * (x - y)) * 3.14159265358979324);
}

/// \brief The manufactured solution used to check every solve.
/// \param[in] x quadrature point
/// \return the manufactured value
__device__ double manufactured(const double x) {
    return exp(x);
}

/// \brief Assembly kernel: one instance per thread, writes its Nystroem
/// system and the manufactured right-hand side into the SoA batch.
/// \param[in]  n quadrature resolution
/// \param[out] A matrix batch, element stride instances
/// \param[out] g right-hand side batch, element stride instances
__global__ void assemble_batch(const int n, double* A, double* g) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= instances) return;

    // Plate separation of the instance, from the sweep ramp.
    const double d = d_min + (d_max - d_min) * t / (instances - 1);
    const double h = 2.0 / (n - 1);

    // Nystroem system on [-1, 1] with the trapezoid rule, written into
    // the SoA batch (element stride instances, coalesced); the
    // manufactured right-hand side is accumulated during the assembly.
    for (int i = 0; i < n; ++i) {
        const double xi = -1.0 + i * h;
        double acc      = 0.0;
        for (int j = 0; j < n; ++j) {
            const double yj = -1.0 + j * h;
            const double wj = (j == 0 || j == n - 1) ? h / 2 : h;
            const double a  = (i == j ? 1.0 : 0.0) + wj * love_kernel(xi, yj, d);
            A[static_cast<std::size_t>(i * n + j) * instances + t] = a;
            acc += a * manufactured(yj);
        }
        g[static_cast<std::size_t>(i) * instances + t] = acc;
    }
}

/// \brief Solve kernel: one instance per thread, factorizes its system
/// in place in the SoA batch, substitutes the manufactured and the
/// physical right-hand sides, and writes the manufactured error and the
/// physical potential.
/// \param[in]     n     quadrature resolution
/// \param[in,out] A     matrix batch, factorized in place
/// \param[out]    piv   pivot batch, element stride instances
/// \param[in,out] g     right-hand side batch, element stride instances
/// \param[out]    u     solution batch, element stride instances
/// \param[out]    err   per-instance error against the manufactured solution
/// \param[out]    u_mid per-instance potential at the central node
/// \param[out]    ok    per-instance success flags
__global__ void solve_batch(const int n, double* A, int* piv, double* g, double* u, double* err,
                            double* u_mid, int* ok) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= instances) return;
    const double h = 2.0 / (n - 1);

    // One factorization, two right-hand sides, every operand walked
    // with the batch stride.
    if (!Solver::factorize(n, A + t, instances, piv + t, instances)) {
        err[t]   = 1.0;
        u_mid[t] = 0.0;
        ok[t]    = 0;
        return;
    }
    Solver::substitute(n, A + t, instances, piv + t, instances, g + t, u + t, instances);
    double e = 0.0;
    for (int i = 0; i < n; ++i)
        e = fmax(e,
                 fabs(u[static_cast<std::size_t>(i) * instances + t] - manufactured(-1.0 + i * h)));
    err[t] = e;

    for (int i = 0; i < n; ++i)
        g[static_cast<std::size_t>(i) * instances + t] = 1.0;
    Solver::substitute(n, A + t, instances, piv + t, instances, g + t, u + t, instances);
    u_mid[t] = u[static_cast<std::size_t>((n - 1) / 2) * instances + t];
    ok[t]    = 1;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // The quadrature resolution comes from outside the program; the
    // bound keeps the materialized batch within two gigabytes of device
    // memory. The default is odd so that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 33;
    if (n < 5 || n > max_n) {
        std::printf("usage: %s [quadrature points in [5, %d]]\n", argv[0], max_n);
        return 1;
    }

    double *d_A = nullptr, *d_g = nullptr, *d_u = nullptr, *d_err = nullptr, *d_umid = nullptr;
    int *d_piv = nullptr, *d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, sizeof(double) * n * n * instances));
    GPU_CHECK(gpuMalloc(&d_g, sizeof(double) * n * instances));
    GPU_CHECK(gpuMalloc(&d_u, sizeof(double) * n * instances));
    GPU_CHECK(gpuMalloc(&d_err, sizeof(double) * instances));
    GPU_CHECK(gpuMalloc(&d_umid, sizeof(double) * instances));
    GPU_CHECK(gpuMalloc(&d_piv, sizeof(int) * n * instances));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * instances));

    // The batch is produced by one kernel and consumed by the other
    // through device memory.
    const int block = 256;
    const int grid  = (instances + block - 1) / block;
    assemble_batch<<<grid, block>>>(n, d_A, d_g);
    GPU_CHECK(gpuGetLastError());
    solve_batch<<<grid, block>>>(n, d_A, d_piv, d_g, d_u, d_err, d_umid, d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> err(instances), u_mid(instances);
    std::vector<int> ok(instances);
    GPU_CHECK(gpuMemcpy(err.data(), d_err, sizeof(double) * err.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(
        gpuMemcpy(u_mid.data(), d_umid, sizeof(double) * u_mid.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(d_A), static_cast<void*>(d_g), static_cast<void*>(d_u),
                    static_cast<void*>(d_err), static_cast<void*>(d_umid),
                    static_cast<void*>(d_piv), static_cast<void*>(d_ok)})
        GPU_CHECK(gpuFree(p));

    int failures = 0;
    double e_max = 0.0, u_lo = 1.0, u_hi = 0.0;
    for (int t = 0; t < instances; ++t) {
        if (!ok[t]) ++failures;
        e_max = std::fmax(e_max, err[t]);
        u_lo  = std::fmin(u_lo, u_mid[t]);
        u_hi  = std::fmax(u_hi, u_mid[t]);
    }

    std::printf("instances = %d, n = %d, manufactured error = %.3e, "
                "u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, e_max, u_mid.front(), d_min, u_mid.back(), d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && e_max < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
