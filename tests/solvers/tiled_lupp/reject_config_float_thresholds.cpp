/// \file
/// \brief Negative compilation test: a configuration whose thresholds
/// are float must be rejected by a double solver.
/// \author Tristan Chenaille
///
/// This translation unit must NOT compile. ctest builds it on purpose
/// and passes only when the compiler emits the scalar-type contract
/// diagnostic of the TiledLUpp solvers.

#include <limits>

#include <tdls/tdls.hpp>

namespace {

struct FloatThresholdsConfig {
    static constexpr int tile_size                    = 3;
    static constexpr tdls::TiledLUppSchedule schedule = tdls::TiledLUppSchedule::RightLooking;
    static constexpr float oot_threshold              = 1e-4f;
    static constexpr float singular_eps               = std::numeric_limits<float>::min();
    static constexpr bool oot_first_acceptable        = true;
    static constexpr bool unroll_inner                = true;
};

} // namespace

int main() {
    double M[81];
    double y[9];
    int piv[9];
    using Solver = tdls::TiledLUppSolverStatic<double, 9, FloatThresholdsConfig>;
    return Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1) ? 0 : 1;
}
