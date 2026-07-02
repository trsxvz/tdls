/// \file
/// \brief Example: reaction substep of an operator-splitting scheme, one
/// independent stiff implicit step chain per cell, parallelized with
/// OpenMP on the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Why the dimension is known at compile time: exactly as in the
/// sequential implicit_ode example, the linear systems are the Newton
/// systems of a Radau IIA step, of size N = stages x species with both
/// factors fixed by the method and by the chemical mechanism. N = 9 is a
/// property of the program, not of the data, so TiledLUppSolverStatic
/// applies.
///
/// Why a batch appears: reactive-transport codes split every global time
/// step into a transport substep and a reaction substep. During the
/// reaction substep each cell integrates its own chemistry independently
/// of every other cell: a moderate number of small stiff systems of
/// identical size, embarrassingly parallel. One omp parallel for
/// distributes the cells; each thread solves on its own stack arrays
/// (local residency), and only the cell states live in shared memory.
///
/// Each cell carries its own temperature factor scaling the reaction
/// rates, so the Newton matrices differ from cell to cell and are
/// generated on the fly from the cell inputs.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include <tdls/tdls.hpp>

namespace {

constexpr int species = 3;                ///< fixed by the Robertson mechanism
constexpr int stages  = 3;                ///< fixed by the Radau IIA method
constexpr int N       = stages * species; ///< Newton system dimension

using Solver = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;

/// \brief Deterministic map from a cell index to a value in [0, 1),
/// standing in for per-cell input data.
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
void robertson_rhs(const double theta, const double* y, double* f) {
    f[0] = theta * (-0.04 * y[0] + 1e4 * y[1] * y[2]);
    f[1] = theta * (0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]);
    f[2] = theta * (3e7 * y[1] * y[1]);
}

/// \brief Analytic Jacobian of the scaled Robertson kinetics, row-major
/// 3 x 3.
/// \param[in]  theta temperature factor of the cell
/// \param[in]  y     species concentrations
/// \param[out] J     derivative of robertson_rhs with respect to y
void robertson_jacobian(const double theta, const double* y, double* J) {
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
bool radau_step(const double (&butcher)[stages][stages], const double theta, const double h,
                double (&y)[species]) {
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
                correction = std::fmax(correction, std::fabs(dz[e]));
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

} // namespace

int main(int argc, char** argv) {
    // Moderate batch: one independent chemistry integration per cell.
    const int cells = argc > 1 ? std::atoi(argv[1]) : 65536;
    if (cells < 1) {
        std::printf("usage: %s [number of cells >= 1]\n", argv[0]);
        return 1;
    }

    const double s6                      = std::sqrt(6.0);
    const double butcher[stages][stages] = {
        {(88 - 7 * s6) / 360, (296 - 169 * s6) / 1800, (-2 + 3 * s6) / 225},
        {(296 + 169 * s6) / 1800, (88 + 7 * s6) / 360, (-2 - 3 * s6) / 225},
        {(16 - s6) / 36, (16 + s6) / 36, 1.0 / 9}};

    const double h  = 1e-3; // time step of the reaction substep
    const int steps = 10;   // integrate each cell to t = 0.01

    // Shared cell states, written once per cell by its owning iteration.
    std::vector<double> state(static_cast<std::size_t>(species) * cells);

    int failures = 0, unphysical = 0;
    double mass_err = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : failures, unphysical)                      \
    reduction(max : mass_err)
    for (int c = 0; c < cells; ++c) {
        // Per-cell inputs: fresh mixture and a temperature factor in
        // [0.5, 2], log-uniform across the batch.
        double y[species]  = {1.0, 0.0, 0.0};
        const double theta = 0.5 * std::pow(4.0, hash01(static_cast<unsigned long long>(c)));

        bool ok = true;
        for (int s = 0; s < steps && ok; ++s)
            ok = radau_step(butcher, theta, h, y);
        if (!ok) ++failures;

        for (int a = 0; a < species; ++a)
            state[static_cast<std::size_t>(species) * c + a] = y[a];

        // The Robertson system conserves the total mass exactly, and the
        // trajectory stays a physical concentration vector.
        mass_err = std::fmax(mass_err, std::fabs(y[0] + y[1] + y[2] - 1.0));
        if (!(y[0] > 0.0 && y[0] < 1.0 && y[1] > 0.0 && y[1] < 1e-3 && y[2] > 0.0 && y[2] < 1.0))
            ++unphysical;
    }

    std::printf("cells = %d, threads = %d, max mass error = %.3e\n", cells, omp_get_max_threads(),
                mass_err);
    return failures == 0 && unphysical == 0 && mass_err < 1e-10 ? 0 : 1;
}
