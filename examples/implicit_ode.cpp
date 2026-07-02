/// \file
/// \brief Example: stiff chemical kinetics integrated with an implicit
/// Runge-Kutta method, using the compile-time TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Why the dimension is known at compile time: the linear systems solved
/// here are the Newton systems of a Radau IIA time step. Their size is
/// N = stages x species, where the number of stages is fixed by the
/// integration method (3 for Radau IIA of order 5) and the number of
/// species is fixed by the chemical mechanism (3 for the Robertson
/// problem). Both are properties of the MODEL and the METHOD, written in
/// the program itself: no input data can change them, so N = 9 is a
/// compile-time constant and TiledLUppSolverStatic applies.
///
/// The example also shows the canonical reason for the split
/// factorize/substitute interface: following the classical RADAU5
/// practice, the Newton matrix is built from the Jacobian FROZEN at the
/// beginning of the step, factorized once, and the factorization is
/// reused by every Newton iteration through substitute().
///
/// Every array lives on the stack of main() (local residency).

#include <cmath>
#include <cstdio>

#include <tdls/tdls.hpp>

namespace {

constexpr int species = 3;                ///< fixed by the Robertson mechanism
constexpr int stages  = 3;                ///< fixed by the Radau IIA method
constexpr int N       = stages * species; ///< Newton system dimension

using Solver = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;

/// \brief Right-hand side of the Robertson kinetics, the classical stiff
/// benchmark: a slow reaction feeding two fast ones.
/// \param[in]  y species concentrations
/// \param[out] f time derivatives
void robertson_rhs(const double* y, double* f) {
    f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    f[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1];
    f[2] = 3e7 * y[1] * y[1];
}

/// \brief Analytic Jacobian of the Robertson kinetics, row-major 3 x 3.
/// \param[in]  y species concentrations
/// \param[out] J derivative of robertson_rhs with respect to y
void robertson_jacobian(const double* y, double* J) {
    J[0] = -0.04;
    J[1] = 1e4 * y[2];
    J[2] = 1e4 * y[1];
    J[3] = 0.04;
    J[4] = -1e4 * y[2] - 6e7 * y[1];
    J[5] = -1e4 * y[1];
    J[6] = 0.0;
    J[7] = 6e7 * y[1];
    J[8] = 0.0;
}

} // namespace

int main() {
    // Butcher matrix of Radau IIA with 3 stages (order 5). The method is
    // stiffly accurate: the solution of the step is the last stage.
    const double s6                      = std::sqrt(6.0);
    const double butcher[stages][stages] = {
        {(88 - 7 * s6) / 360, (296 - 169 * s6) / 1800, (-2 + 3 * s6) / 225},
        {(296 + 169 * s6) / 1800, (88 + 7 * s6) / 360, (-2 - 3 * s6) / 225},
        {(16 - s6) / 36, (16 + s6) / 36, 1.0 / 9}};

    double y[species] = {1.0, 0.0, 0.0}; // initial concentrations
    const double h    = 1e-3;            // time step
    const int steps   = 1000;            // integrate to t = 1

    for (int step = 0; step < steps; ++step) {
        // Newton matrix with the Jacobian frozen at the beginning of the
        // step: M[(i,a),(j,b)] = delta_ij delta_ab - h butcher[i][j] J0[a][b],
        // constant during the whole step by construction.
        double J0[species * species];
        robertson_jacobian(y, J0);
        double M[N * N];
        for (int i = 0; i < stages; ++i)
            for (int j = 0; j < stages; ++j)
                for (int a = 0; a < species; ++a)
                    for (int b = 0; b < species; ++b)
                        M[(i * species + a) * N + (j * species + b)] =
                            (i == j && a == b ? 1.0 : 0.0) -
                            h * butcher[i][j] * J0[a * species + b];

        // Factorize ONCE per step; every Newton iteration below reuses
        // the factorization through substitute().
        int piv[N];
        if (!Solver::factorize<true, true>(M, 1, piv, 1)) {
            std::printf("singular Newton matrix at step %d\n", step);
            return 1;
        }

        // Newton iteration on the stacked stage values Z = (Y1, Y2, Y3),
        // started from the current state.
        double Z[N];
        for (int i = 0; i < stages; ++i)
            for (int a = 0; a < species; ++a)
                Z[i * species + a] = y[a];

        double correction = 1.0;
        for (int iter = 0; iter < 50 && correction > 1e-13; ++iter) {
            // Residual R_i = Y_i - y - h sum_j butcher[i][j] f(Y_j).
            double f[stages][species];
            for (int j = 0; j < stages; ++j)
                robertson_rhs(&Z[j * species], f[j]);
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
        if (correction > 1e-13) {
            std::printf("Newton did not converge at step %d\n", step);
            return 1;
        }

        // Stiffly accurate method: the new state is the last stage.
        for (int a = 0; a < species; ++a)
            y[a] = Z[(stages - 1) * species + a];
    }

    // Self-checks: the Robertson system conserves the total mass exactly,
    // and the trajectory stays a physical concentration vector.
    const double mass = y[0] + y[1] + y[2];
    std::printf("t = 1: y = (%.6f, %.6e, %.6f), mass = %.15f\n", y[0], y[1], y[2], mass);
    const bool conserved = std::fabs(mass - 1.0) < 1e-10;
    const bool physical =
        y[0] > 0.0 && y[0] < 1.0 && y[1] > 0.0 && y[1] < 1e-3 && y[2] > 0.0 && y[2] < 1.0;
    return conserved && physical ? 0 : 1;
}
