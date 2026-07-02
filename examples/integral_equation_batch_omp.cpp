/// \file
/// \brief Example: parameter sweep of Love's integral equation, one
/// dense Nystroem system per plate separation, parallelized with OpenMP
/// on the runtime TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Companion of integral_equation.cpp at the next scale: the capacitor
/// potential is computed for a whole sweep of plate separations d, the
/// canonical embarrassingly parallel workload of parametric studies.
/// Every instance assembles and solves its own dense n x n Nystroem
/// system; one omp parallel for distributes the instances and each
/// thread reuses its own buffers.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched (here the command line), so the dimension is a runtime
/// value and TiledLUppSolverDynamic applies.
///
/// Each instance factorizes once and substitutes two right-hand sides:
/// a manufactured one (exact self-check at solver accuracy) and the
/// physical unit potential, giving the curve of the potential against
/// the plate separation.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <omp.h>

#include <tdls/tdls.hpp>

namespace {

using Solver = tdls::TiledLUppSolverDynamic<double>;

constexpr int instances = 20000; ///< plate separations in the sweep
constexpr double d_min  = 0.5;   ///< smallest plate separation
constexpr double d_max  = 2.0;   ///< largest plate separation

/// \brief Love kernel of the parallel-plate capacitor.
/// \param[in] x first quadrature point
/// \param[in] y second quadrature point
/// \param[in] d plate separation
/// \return the kernel value
double love_kernel(const double x, const double y, const double d) {
    return d / ((d * d + (x - y) * (x - y)) * 3.14159265358979324);
}

/// \brief The manufactured solution used to check every solve.
/// \param[in] x quadrature point
/// \return the manufactured value
double manufactured(const double x) {
    return std::exp(x);
}

} // namespace

int main(int argc, char** argv) {
    // The quadrature resolution comes from outside the program; the
    // default is odd so that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 41;
    if (n < 5 || n > 100) {
        std::printf("usage: %s [quadrature points in [5, 100]]\n", argv[0]);
        return 1;
    }
    const double h = 2.0 / (n - 1);
    const int mid  = (n - 1) / 2;

    int failures = 0;
    double err = 0.0, u_first = 0.0, u_last = 0.0;
    double u_lo = 1.0, u_hi = 0.0;
#pragma omp parallel reduction(+ : failures) reduction(max : err, u_hi) reduction(min : u_lo)
    {
        // Thread-local buffers, reused across the instances of the
        // thread.
        std::vector<double> A(static_cast<std::size_t>(n) * n), g(n), u(n);
        std::vector<int> piv(n);

#pragma omp for schedule(static)
        for (int c = 0; c < instances; ++c) {
            // Plate separation of the instance, from the sweep ramp.
            const double d = d_min + (d_max - d_min) * c / (instances - 1);

            // Nystroem system on [-1, 1] with the trapezoid rule; the
            // manufactured right-hand side is accumulated during the
            // assembly.
            for (int i = 0; i < n; ++i) {
                const double xi = -1.0 + i * h;
                double acc      = 0.0;
                for (int j = 0; j < n; ++j) {
                    const double yj = -1.0 + j * h;
                    const double wj = (j == 0 || j == n - 1) ? h / 2 : h;
                    const double a  = (i == j ? 1.0 : 0.0) + wj * love_kernel(xi, yj, d);
                    A[static_cast<std::size_t>(i) * n + j] = a;
                    acc += a * manufactured(yj);
                }
                g[i] = acc;
            }

            // One factorization, two right-hand sides.
            if (!Solver::factorize(n, A.data(), 1, piv.data(), 1)) {
                ++failures;
                continue;
            }
            Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
            for (int i = 0; i < n; ++i)
                err = std::fmax(err, std::fabs(u[i] - manufactured(-1.0 + i * h)));

            std::fill(g.begin(), g.end(), 1.0);
            Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
            u_lo = std::fmin(u_lo, u[mid]);
            u_hi = std::fmax(u_hi, u[mid]);
            if (c == 0) u_first = u[mid];
            if (c == instances - 1) u_last = u[mid];
        }
    }

    std::printf("instances = %d, n = %d, threads = %d, manufactured error = %.3e, "
                "u(0): %.6f at d = %.1f -> %.6f at d = %.1f\n",
                instances, n, omp_get_max_threads(), err, u_first, d_min, u_last, d_max);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return failures == 0 && err < 1e-12 && u_lo > 0.4 && u_hi < 1.0 ? 0 : 1;
}
