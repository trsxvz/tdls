/// \file
/// \brief Example: reaction substep of an operator-splitting scheme on
/// GPU, one independent stiff implicit step chain per cell, with the
/// linear systems held in registers through the internal residencies of
/// the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential and OpenMP implicit_ode examples, the Newton systems of a
/// Radau IIA step have size N = stages x species, fixed by the method
/// and by the chemical mechanism. N = 9 is a property of the program.
///
/// Why the residencies are internal: each thread passes
/// residency booleans set to true, so the solver indexes the matrix,
/// the pivots and the right-hand side with compile-time strides and the
/// whole system lives in thread registers. Device memory only holds the
/// per-cell inputs (a temperature factor) and outputs (the cell state):
/// the batch of matrices is never materialized, so the batch size is
/// bounded by compute capacity, not by device memory.

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

/// \brief One Radau IIA step with a frozen Jacobian: the Newton matrix
/// is factorized once and the factorization is reused by every Newton
/// iteration. The Jacobian is frozen at the step start and, following
/// stiff integrator practice, refreshed at the last stage when Newton
/// stalls (strong transients at high temperature factors).
/// \param[in]     butcher Butcher matrix of the method
/// \param[in]     theta   temperature factor of the cell
/// \param[in]     h       time step
/// \param[in,out] y       cell state, advanced by h on success
/// \return true when the factorizations succeeded and Newton converged
__device__ bool radau_step(const double (&butcher)[stages][stages], const double theta,
                           const double h, double (&y)[species]) {
    // Newton iterates on the stacked stage values Z = (Y1, Y2, Y3),
    // started from the current state and kept across refreshes.
    double Z[N];
    for (int i = 0; i < stages; ++i)
        for (int a = 0; a < species; ++a)
            Z[i * species + a] = y[a];

    for (int attempt = 0; attempt < 3; ++attempt) {
        double J0[species * species];
        robertson_jacobian(theta, attempt == 0 ? y : &Z[(stages - 1) * species], J0);
        double M[N * N];
        for (int i = 0; i < stages; ++i)
            for (int j = 0; j < stages; ++j)
                for (int a = 0; a < species; ++a)
                    for (int b = 0; b < species; ++b)
                        M[(i * species + a) * N + (j * species + b)] =
                            (i == j && a == b ? 1.0 : 0.0) -
                            h * butcher[i][j] * J0[a * species + b];

        int piv[N];
        if (!Solver::factorize<true, true>(M, 1, piv, 1)) return false;

        double correction = 1.0;
        for (int iter = 0; iter < 30 && correction > 1e-12; ++iter) {
            double f[stages][species];
            for (int j = 0; j < stages; ++j)
                robertson_rhs(theta, &Z[j * species], f[j]);
            double r[N], dz[N];
            for (int i = 0; i < stages; ++i)
                for (int a = 0; a < species; ++a) {
                    double acc = Z[i * species + a] - y[a];
                    for (int j = 0; j < stages; ++j)
                        acc -= h * butcher[i][j] * f[j][a];
                    r[i * species + a] = -acc;
                }
            Solver::substitute<true, true, true>(M, 1, piv, 1, r, dz, 1);
            correction = 0.0;
            for (int e = 0; e < N; ++e) {
                Z[e] += dz[e];
                correction = fmax(correction, fabs(dz[e]));
            }
        }
        if (correction <= 1e-12) {
            // Stiffly accurate method: the new state is the last stage.
            for (int a = 0; a < species; ++a)
                y[a] = Z[(stages - 1) * species + a];
            return true;
        }
    }
    return false;
}

/// \brief One cell per thread: reads the cell inputs, integrates the
/// chemistry entirely in registers, writes the final state (SoA layout,
/// coalesced) and a success flag.
/// \param[in]  cells number of cells
/// \param[in]  steps number of Radau steps per cell
/// \param[in]  h     time step
/// \param[in]  theta per-cell temperature factors
/// \param[out] state final states, species-major SoA of extent cells
/// \param[out] ok    per-cell success flags
__global__ void reaction_substep(const int cells, const int steps, const double h,
                                 const double* theta, double* state, int* ok) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= cells) return;

    const double s6                      = sqrt(6.0);
    const double butcher[stages][stages] = {
        {(88 - 7 * s6) / 360, (296 - 169 * s6) / 1800, (-2 + 3 * s6) / 225},
        {(296 + 169 * s6) / 1800, (88 + 7 * s6) / 360, (-2 - 3 * s6) / 225},
        {(16 - s6) / 36, (16 + s6) / 36, 1.0 / 9}};

    // Per-cell inputs: fresh mixture and the cell temperature factor.
    double y[species] = {1.0, 0.0, 0.0};

    bool success = true;
    for (int s = 0; s < steps && success; ++s)
        success = radau_step(butcher, theta[t], h, y);

    for (int a = 0; a < species; ++a)
        state[static_cast<std::size_t>(a) * cells + t] = y[a];
    ok[t] = success ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (!gpu_device_available()) {
        std::printf("no device available, skipping\n");
        return gpu_skip_code;
    }

    // Large batch: device memory only holds 4 doubles per cell, the
    // systems themselves never leave the registers.
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

    double *d_theta = nullptr, *d_state = nullptr;
    int* d_ok = nullptr;
    GPU_CHECK(gpuMalloc(&d_theta, sizeof(double) * cells));
    GPU_CHECK(gpuMalloc(&d_state, sizeof(double) * species * cells));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * cells));
    GPU_CHECK(
        gpuMemcpy(d_theta, theta.data(), sizeof(double) * theta.size(), gpuMemcpyHostToDevice));

    const int block = 256;
    reaction_substep<<<(cells + block - 1) / block, block>>>(cells, steps, h, d_theta, d_state,
                                                             d_ok);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<double> state(static_cast<std::size_t>(species) * cells);
    std::vector<int> ok(cells);
    GPU_CHECK(
        gpuMemcpy(state.data(), d_state, sizeof(double) * state.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ok.data(), d_ok, sizeof(int) * ok.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_theta));
    GPU_CHECK(gpuFree(d_state));
    GPU_CHECK(gpuFree(d_ok));

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
