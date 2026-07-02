/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// GPU, with the linear systems materialized in device memory in SoA
/// layout and solved through the external residencies of the
/// compile-time TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Companion of implicit_ode_batch_gpu.cu: same physics, same method,
/// opposite residency choice. Here the residency booleans are false,
/// so the matrices, pivots and right-hand sides live in device memory
/// and the solver walks them with a runtime element stride. The batch
/// is stored species-of-arrays: element k of system t sits at
/// base[k * cells + t], so consecutive threads touch consecutive
/// addresses and every access is coalesced.
///
/// External residency is the right choice when the systems cannot or
/// should not stay in registers: larger dimensions, register pressure
/// throttling occupancy, or matrices produced by a separate assembly
/// kernel, the pattern shown here (one kernel assembles the batch,
/// another consumes it). The price is that the batch now occupies
/// device memory, which bounds its size; the register variant has no
/// such bound.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tdls/tdls.hpp>

#include "gpu_runtime.hpp"

namespace {

constexpr int species = 3;                ///< fixed by the Robertson mechanism
constexpr int stages  = 3;                ///< fixed by the Radau IIA method
constexpr int N       = stages * species; ///< Newton system dimension

using Solver = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;

/// \brief Deterministic map from a cell index to a value in [0, 1),
/// used on the host to synthesize the per-cell input data.
/// \param[in] i cell index
/// \return value in [0, 1)
double hash01(const unsigned long long i) {
    unsigned long long z = i + 0x9e3779b97f4a7c15ull;
    z                    = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z                    = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z                    = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

/// \brief Fills the Butcher matrix of Radau IIA with 3 stages (order 5).
/// \param[out] butcher Butcher matrix of the method
__device__ void radau_butcher(double (&butcher)[stages][stages]) {
    const double s6 = sqrt(6.0);
    butcher[0][0]   = (88 - 7 * s6) / 360;
    butcher[0][1]   = (296 - 169 * s6) / 1800;
    butcher[0][2]   = (-2 + 3 * s6) / 225;
    butcher[1][0]   = (296 + 169 * s6) / 1800;
    butcher[1][1]   = (88 + 7 * s6) / 360;
    butcher[1][2]   = (-2 - 3 * s6) / 225;
    butcher[2][0]   = (16 - s6) / 36;
    butcher[2][1]   = (16 + s6) / 36;
    butcher[2][2]   = 1.0 / 9;
}

/// \brief Right-hand side of the Robertson kinetics with a temperature
/// factor scaling all reaction rates.
/// \param[in]  theta temperature factor of the cell
/// \param[in]  y     species concentrations
/// \param[out] f     time derivatives
__device__ void robertson_rhs(const double theta, const double* y, double* f) {
    f[0] = theta * (-0.04 * y[0] + 1e4 * y[1] * y[2]);
    f[1] = theta * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]);
    f[2] = theta * (3e7 * y[1] * y[1]);
}

/// \brief Analytic Jacobian of the scaled Robertson kinetics, row-major
/// 3 x 3.
/// \param[in]  theta temperature factor of the cell
/// \param[in]  y     species concentrations
/// \param[out] J     derivative of robertson_rhs with respect to y
__device__ void robertson_jacobian(const double theta, const double* y, double* J) {
    J[0] = theta * -0.04;
    J[1] = theta * 1e4 * y[2];
    J[2] = theta * 1e4 * y[1];
    J[3] = theta * 0.04;
    J[4] = theta * (-1e4 * y[2] - 6e7 * y[1]);
    J[5] = theta * -1e4 * y[1];
    J[6] = 0.0;
    J[7] = theta * 6e7 * y[1];
    J[8] = 0.0;
}

/// \brief Writes the Newton matrix of a Radau IIA step into device
/// memory with a runtime element stride.
/// \param[in]  butcher Butcher matrix of the method
/// \param[in]  theta   temperature factor of the cell
/// \param[in]  y       state the Jacobian is frozen at
/// \param[in]  h       time step
/// \param[out] M       matrix of the cell, element stride `stride`
/// \param[in]  stride  element stride of M
__device__ void assemble_newton_matrix(const double (&butcher)[stages][stages], const double theta,
                                       const double* y, const double h, double* M,
                                       const int stride) {
    double J0[species * species];
    robertson_jacobian(theta, y, J0);
    for (int i = 0; i < stages; ++i)
        for (int j = 0; j < stages; ++j)
            for (int a = 0; a < species; ++a)
                for (int b = 0; b < species; ++b)
                    M[static_cast<std::size_t>((i * species + a) * N + (j * species + b)) *
                      stride] =
                        (i == j && a == b ? 1.0 : 0.0) - h * butcher[i][j] * J0[a * species + b];
}

/// \brief Assembly kernel: one cell per thread, freezes the Jacobian at
/// the current state and writes the Newton matrix into the SoA batch.
/// \param[in]  cells number of cells
/// \param[in]  h     time step
/// \param[in]  theta per-cell temperature factors
/// \param[in]  state current states, species-major SoA of extent cells
/// \param[out] M     matrix batch, element stride cells
__global__ void assemble_batch(const int cells, const double h, const double* theta,
                               const double* state, double* M) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= cells) return;

    double butcher[stages][stages];
    radau_butcher(butcher);
    double y[species];
    for (int a = 0; a < species; ++a)
        y[a] = state[static_cast<std::size_t>(a) * cells + t];
    assemble_newton_matrix(butcher, theta[t], y, h, M + t, cells);
}

/// \brief Solve kernel: one cell per thread, factorizes the assembled
/// matrix in place in device memory and runs the Newton iteration with
/// every solver operand in the SoA batch (external residencies). When
/// Newton stalls the matrix is reassembled at the last stage and
/// refactorized, following stiff integrator practice.
/// \param[in]     cells number of cells
/// \param[in]     h     time step
/// \param[in]     theta per-cell temperature factors
/// \param[in,out] state cell states, advanced by h on success
/// \param[in,out] M     matrix batch, factorized in place
/// \param[out]    piv   pivot batch, element stride cells
/// \param[out]    r     residual batch, element stride cells
/// \param[out]    dz    correction batch, element stride cells
/// \param[out]    ok    per-cell success flags
__global__ void solve_batch(const int cells, const double h, const double* theta, double* state,
                            double* M, int* piv, double* r, double* dz, int* ok) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= cells) return;

    double butcher[stages][stages];
    radau_butcher(butcher);
    const double theta_t = theta[t];
    double y[species];
    for (int a = 0; a < species; ++a)
        y[a] = state[static_cast<std::size_t>(a) * cells + t];

    // Newton iterates on the stacked stage values Z = (Y1, Y2, Y3); the
    // state fits in registers, the linear algebra stays in the batch.
    double Z[N];
    for (int i = 0; i < stages; ++i)
        for (int a = 0; a < species; ++a)
            Z[i * species + a] = y[a];

    bool converged = false;
    for (int attempt = 0; attempt < 3 && !converged; ++attempt) {
        if (attempt > 0)
            assemble_newton_matrix(butcher, theta_t, &Z[(stages - 1) * species], h, M + t, cells);
        if (!Solver::factorize<false, false>(M + t, cells, piv + t, cells)) break;

        double correction = 1.0;
        for (int iter = 0; iter < 30 && correction > 1e-12; ++iter) {
            double f[stages][species];
            for (int j = 0; j < stages; ++j)
                robertson_rhs(theta_t, &Z[j * species], f[j]);
            for (int i = 0; i < stages; ++i)
                for (int a = 0; a < species; ++a) {
                    double acc = Z[i * species + a] - y[a];
                    for (int j = 0; j < stages; ++j)
                        acc -= h * butcher[i][j] * f[j][a];
                    r[static_cast<std::size_t>(i * species + a) * cells + t] = -acc;
                }
            Solver::substitute<false, false, false>(M + t, cells, piv + t, cells, r + t, dz + t,
                                                    cells);
            correction = 0.0;
            for (int e = 0; e < N; ++e) {
                const double d = dz[static_cast<std::size_t>(e) * cells + t];
                Z[e] += d;
                correction = fmax(correction, fabs(d));
            }
        }
        converged = correction <= 1e-12;
    }

    if (converged) {
        // Stiffly accurate method: the new state is the last stage.
        for (int a = 0; a < species; ++a)
            state[static_cast<std::size_t>(a) * cells + t] = Z[(stages - 1) * species + a];
    }
    ok[t] = converged ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // The batch is materialized in device memory (about 850 bytes per
    // cell), so its size is bounded by the device capacity.
    const int cells = argc > 1 ? std::atoi(argv[1]) : 1 << 21;
    if (cells < 1) {
        std::printf("usage: %s [number of cells >= 1]\n", argv[0]);
        return 1;
    }
    const double h  = 1e-3; // time step of the reaction substep
    const int steps = 2;    // integrate each cell to t = 0.002

    // Per-cell temperature factors in [0.5, 2], log-uniform across the
    // batch, synthesized on the host as the input data of the substep.
    std::vector<double> theta(cells);
    for (int c = 0; c < cells; ++c)
        theta[c] = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

    double *d_theta = nullptr, *d_state = nullptr, *d_M = nullptr, *d_r = nullptr, *d_dz = nullptr;
    int *d_piv = nullptr, *d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_theta, sizeof(double) * cells));
    GPU_CHECK(
        gpuMemcpy(d_theta, theta.data(), sizeof(double) * theta.size(), gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * species * cells));
    GPU_CHECK(gpuMalloc(&d_M, sizeof(double) * N * N * cells));
    GPU_CHECK(gpuMalloc(&d_r, sizeof(double) * N * cells));
    GPU_CHECK(gpuMalloc(&d_dz, sizeof(double) * N * cells));
    GPU_CHECK(gpuMalloc(&d_piv, sizeof(int) * N * cells));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * cells));

    // Fresh mixture in every cell.
    std::vector<double> state(static_cast<std::size_t>(species) * cells, 0.0);
    std::fill(state.begin(), state.begin() + cells, 1.0);
    GPU_CHECK(
        gpuMemcpy(d_state, state.data(), sizeof(double) * state.size(), gpuMemcpyHostToDevice));

    // One assembly and one solve kernel per step: the batch is produced
    // by one kernel and consumed by the other through device memory.
    const int block = 256;
    const int grid  = (cells + block - 1) / block;
    for (int s = 0; s < steps; ++s) {
        assemble_batch<<<grid, block>>>(cells, h, d_theta, d_state, d_M);
        GPU_CHECK(gpuGetLastError());
        solve_batch<<<grid, block>>>(cells, h, d_theta, d_state, d_M, d_piv, d_r, d_dz, d_ok);
        GPU_CHECK(gpuGetLastError());
    }
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<int> ok(cells);
    GPU_CHECK(
        gpuMemcpy(state.data(), d_state, sizeof(double) * state.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    for (void* p : {static_cast<void*>(d_theta), static_cast<void*>(d_state),
                    static_cast<void*>(d_M), static_cast<void*>(d_r), static_cast<void*>(d_dz),
                    static_cast<void*>(d_piv), static_cast<void*>(d_ok)})
        GPU_CHECK(gpuFree(p));

    // The Robertson system conserves the total mass exactly, and the
    // trajectory stays a physical concentration vector.
    int failures = 0, unphysical = 0;
    double mass_err = 0.0;
    for (int c = 0; c < cells; ++c) {
        if (!ok[c]) ++failures;
        const double y0 = state[c];
        const double y1 = state[static_cast<std::size_t>(cells) + c];
        const double y2 = state[static_cast<std::size_t>(2) * cells + c];
        mass_err        = std::fmax(mass_err, std::fabs(y0 + y1 + y2 - 1.0));
        if (!(y0 > 0.0 && y0 < 1.0 && y1 > 0.0 && y1 < 1e-3 && y2 > 0.0 && y2 < 1.0)) ++unphysical;
    }

    std::printf("cells = %d, max mass error = %.3e\n", cells, mass_err);
    return failures == 0 && unphysical == 0 && mass_err < 1e-10 ? 0 : 1;
}
