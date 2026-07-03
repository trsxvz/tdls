/// \file
/// \brief Negative compilation test: a configuration whose singularity
/// floor exceeds the out-of-tile threshold must be rejected.
/// \author Tristan Chenaille
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the threshold-ordering
/// contract diagnostic of the TiledLUpp solvers (the floor only guards
/// the out-of-tile recovery path, so it would be unreachable above the
/// threshold).

#include <tdls/tdls.hpp>

namespace {

struct EpsAboveThresholdConfig {
    static constexpr int tile_size                    = 3;
    static constexpr tdls::TiledLUppSchedule schedule = tdls::TiledLUppSchedule::RightLooking;
    static constexpr double oot_threshold             = 1e-10;
    static constexpr double singular_eps              = 1e-4;
    static constexpr bool oot_first_acceptable        = true;
    static constexpr bool unroll_inner                = true;
};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, EpsAboveThresholdConfig>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
