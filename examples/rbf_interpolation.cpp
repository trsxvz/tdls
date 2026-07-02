/// \file
/// \brief Example: radial-basis-function interpolation of scattered
/// measurements, using the runtime TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Why the dimension is only known at run time: the linear system solved
/// here couples one weight per MEASUREMENT POINT, and the number of
/// measurements arrives with the data (a file, an acquisition, here the
/// command line). No rebuild can be expected when a new campaign brings
/// 97 points instead of 120, so the dimension is a runtime value and
/// TiledLUppSolverDynamic applies.
///
/// The kernel matrix K[i][j] = phi(|x_i - x_j|) is dense by nature
/// (every measurement interacts with every other), which is exactly the
/// shape a dense direct solver is meant for. This is the workhorse of
/// kriging, meshless methods and surrogate models.
///
/// Storage is plain contiguous memory (stride 1 everywhere); the runtime
/// solver has no residency parameters.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tdls/tdls.hpp>

namespace {

using Solver = tdls::TiledLUppSolverDynamic<double>;

/// \brief Multiquadric kernel, a standard choice for scattered-data
/// interpolation.
/// \param[in] r     distance between two points
/// \param[in] shape kernel shape parameter
/// \return the kernel value
double multiquadric(const double r, const double shape) {
    return std::sqrt(1.0 + shape * shape * r * r);
}

/// \brief The field being measured, used to synthesize the data set and
/// to judge the interpolant away from the measurement points.
/// \param[in] x position in [0, 1]
/// \return the field value
double field(const double x) {
    return std::sin(6.28318530717958648 * x) * std::exp(-x);
}

} // namespace

int main(int argc, char** argv) {
    // The size of the campaign comes from outside the program.
    const int n = argc > 1 ? std::atoi(argv[1]) : 120;
    if (n < 2) {
        std::printf("usage: %s [number of measurement points >= 2]\n", argv[0]);
        return 1;
    }
    const double shape = 2.0;

    // Synthetic measurement campaign standing in for a data file: n
    // sample positions and the measured field values.
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / (n - 1);
        y[i] = field(x[i]);
    }

    // Dense kernel matrix, one row and one column per measurement.
    std::vector<double> K(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            K[static_cast<std::size_t>(i) * n + j] = multiquadric(x[i] - x[j], shape);
    std::vector<double> K0(K); // kept to judge the solve quality

    // Interpolation weights: solve K w = y at the runtime dimension n.
    std::vector<double> w(n);
    std::vector<int> piv(n);
    if (!Solver::solve(n, K.data(), 1, piv.data(), 1, y.data(), w.data(), 1)) {
        std::printf("singular kernel matrix\n");
        return 1;
    }

    // Self-check on the normwise backward error. Smooth radial kernels
    // are ill-conditioned, so the raw residual legitimately grows with
    // the weights; the backward error is the conditioning-independent
    // measure of the solve, and stays at rounding level.
    double residual = 0.0, k_max = 0.0, w_max = 0.0, y_max = 0.0;
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            acc += K0[static_cast<std::size_t>(i) * n + j] * w[j];
            k_max = std::fmax(k_max, std::fabs(K0[static_cast<std::size_t>(i) * n + j]));
        }
        residual = std::fmax(residual, std::fabs(acc - y[i]));
        w_max    = std::fmax(w_max, std::fabs(w[i]));
        y_max    = std::fmax(y_max, std::fabs(y[i]));
    }
    const double backward_error = residual / (k_max * w_max + y_max);

    // Informative only: interpolation error away from the measurements.
    double error = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        const double xm = 0.5 * (x[i] + x[i + 1]);
        double s        = 0.0;
        for (int j = 0; j < n; ++j)
            s += w[j] * multiquadric(xm - x[j], shape);
        error = std::fmax(error, std::fabs(s - field(xm)));
    }

    std::printf("n = %d, backward error = %.3e, midpoint interpolation error = %.3e\n", n,
                backward_error, error);
    return backward_error < 1e-12 ? 0 : 1;
}
