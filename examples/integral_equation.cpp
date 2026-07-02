/// \file
/// \brief Example: Love's integral equation solved by the Nystroem
/// method, using the runtime TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Love's equation gives the potential of a circular parallel-plate
/// capacitor with plate separation d:
///
///   u(x) + (1/pi) integral of d / (d^2 + (x-y)^2) u(y) dy = g(x)
///
/// The Nystroem method replaces the integral by a quadrature over n
/// points, which couples every point to every other: the linear system
/// is dense by nature, exactly what a dense direct solver is for.
///
/// Why the dimension is only known at run time: n is the quadrature
/// resolution, an accuracy versus cost knob chosen when the computation
/// is launched (here the command line). The same binary must serve
/// n = 64 and n = 200 without rebuilding, so the dimension is a runtime
/// value and TiledLUppSolverDynamic applies.
///
/// The factorization is computed once and reused by two right-hand
/// sides: a manufactured one (the discrete operator applied to a known
/// solution, which the solve must return to solver accuracy, whatever
/// the resolution) and the physical unit potential.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tdls/tdls.hpp>

namespace {

using Solver = tdls::TiledLUppSolverDynamic<double>;

/// \brief Love kernel of the parallel-plate capacitor.
/// \param[in] x first quadrature point
/// \param[in] y second quadrature point
/// \param[in] d plate separation
/// \return the kernel value
double love_kernel(const double x, const double y, const double d) {
    return d / ((d * d + (x - y) * (x - y)) * 3.14159265358979324);
}

/// \brief The manufactured solution used to check the solve.
/// \param[in] x quadrature point
/// \return the manufactured value
double manufactured(const double x) {
    return std::exp(x);
}

} // namespace

int main(int argc, char** argv) {
    // The quadrature resolution comes from outside the program; the
    // default is odd so that x = 0 is a node.
    const int n = argc > 1 ? std::atoi(argv[1]) : 121;
    if (n < 5 || n > 1000) {
        std::printf("usage: %s [quadrature points in [5, 1000]]\n", argv[0]);
        return 1;
    }
    const double d = 1.0; // plate separation

    // Nystroem system on [-1, 1] with the trapezoid rule: the identity
    // plus the discretized integral operator. The manufactured
    // right-hand side is the discrete operator applied to a known
    // solution, accumulated during the assembly.
    const double h = 2.0 / (n - 1);
    std::vector<double> A(static_cast<std::size_t>(n) * n), g(n), u(n);
    std::vector<int> piv(n);
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
        std::printf("singular Nystroem matrix\n");
        return 1;
    }

    // Manufactured right-hand side: the solve must return the
    // manufactured solution to solver accuracy (the equation is well
    // conditioned), whatever the quadrature resolution.
    Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        err = std::fmax(err, std::fabs(u[i] - manufactured(-1.0 + i * h)));

    // Physical right-hand side: unit potential on the plate.
    std::fill(g.begin(), g.end(), 1.0);
    Solver::substitute(n, A.data(), 1, piv.data(), 1, g.data(), u.data(), 1);
    const int mid      = (n - 1) / 2;
    const double u_mid = u[mid];

    std::printf("n = %d, manufactured error = %.3e, u(%.2f) = %.6f (d = %.1f)\n", n, err,
                -1.0 + mid * h, u_mid, d);

    // The potential of the unit problem is bounded by construction: the
    // integral operator is positive with norm below one.
    return err < 1e-12 && u_mid > 0.4 && u_mid < 1.0 ? 0 : 1;
}
