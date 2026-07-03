/// \file
/// \brief Negative compilation test: a configuration whose thresholds
/// are double must be rejected by a float solver.
/// \author Tristan Chenaille
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the scalar-type contract
/// diagnostic of the TiledLUpp solvers.

#include <limits>

#include <tdls/tdls.hpp>

namespace {

struct DoubleThresholdsConfig {
    static constexpr int tile_size                    = 3;
    static constexpr tdls::TiledLUppSchedule schedule = tdls::TiledLUppSchedule::RightLooking;
    static constexpr double oot_threshold             = 1e-10;
    static constexpr double singular_eps              = std::numeric_limits<double>::min();
    static constexpr bool oot_first_acceptable        = true;
    static constexpr bool unroll_inner                = true;
};

} // namespace

int main() {
    float A[81];
    float y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverDynamic<float, DoubleThresholdsConfig>;
    return Solver::solve_inplace(9, A, 1, piv, 1, y, 1) ? 0 : 1;
}
