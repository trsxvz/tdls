/// \file
/// \brief Anchor suite of the dynamic TiledLUpp solver.
/// \author Tristan Chenaille
///
/// TiledLUppSolverDynamic is compared against the independent reference LU
/// on the same shape grid as the static anchor, plus the shapes only the
/// dynamic solver accepts: tile size exceeding the dimension (single
/// partial tile) and dimensions beyond the static test grid. The verdict
/// of every case is the normwise backward error of both solvers, plus the
/// exact agreement of the singularity verdicts (one structurally singular
/// system is injected in every batch).

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Runs one anchor comparison on a reproducible batch: solves with
/// the dynamic TiledLUpp solver (contiguous storage, runtime dimension) and with
/// the reference, requires identical verdicts, exactly one singular
/// report on each side, and both backward errors under the tolerance.
/// \tparam T     scalar type
/// \tparam TS    tile size
/// \tparam Sched elimination schedule
/// \param[in] n         system dimension (runtime)
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int TS, tdls::TiledLUppSchedule Sched>
void anchor_case(const int n, const int count, const double bound, const double tolerance,
                 const std::uint64_t seed) {
    using Solver = tdls::TiledLUppSolverDynamic<T, tdls::TiledLUppConfig<T, TS, Sched>>;
    auto batch   = tdls_tests::make_batch<T>(n, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree    = true;
    int singular_tiled     = 0;
    int singular_reference = 0;
    double be_tiled        = 0.0;
    double be_reference    = 0.0;

    std::vector<T> At(static_cast<std::size_t>(n) * n), Ar(static_cast<std::size_t>(n) * n);
    std::vector<T> x(n), xr(n);
    std::vector<int> piv(n), pivr(n);
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + n * n, At.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + n * n, Ar.begin());
        const bool ok_tiled =
            Solver::solve(n, At.data(), 1, piv.data(), 1, batch.rhs(s), x.data(), 1);
        const bool ok_reference =
            tdls_tests::reference_solve(Ar.data(), pivr.data(), batch.rhs(s), xr.data(), n);
        if (ok_tiled != ok_reference) verdicts_agree = false;
        if (!ok_tiled) ++singular_tiled;
        if (!ok_reference) ++singular_reference;
        if (ok_tiled && ok_reference) {
            be_tiled = std::max(
                be_tiled, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), n));
            be_reference =
                std::max(be_reference,
                         tdls_tests::backward_error(batch.matrix(s), xr.data(), batch.rhs(s), n));
        }
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(singular_tiled == 1);
    TDLS_CHECK(singular_reference == 1);
    TDLS_CHECK_LE(be_tiled, tolerance);
    TDLS_CHECK_LE(be_reference, tolerance);
}

} // namespace

/// Emits the RL and LL anchor cases of one (type, n, TS, regime) cell.
#define TDLS_ANCHOR_CASES(T, N, TS, REGIME, COUNT, BOUND, TOL, SEED)                               \
    TDLS_TEST_CASE("tiledlupp/oracle/dynamic/" #T "/n=" #N ",TS=" #TS ",RL," REGIME) {             \
        anchor_case<T, TS, tdls::TiledLUppSchedule::RightLooking>(N, COUNT, BOUND, TOL, SEED);     \
    }                                                                                              \
    TDLS_TEST_CASE("tiledlupp/oracle/dynamic/" #T "/n=" #N ",TS=" #TS ",LL," REGIME) {             \
        anchor_case<T, TS, tdls::TiledLUppSchedule::LeftLooking>(N, COUNT, BOUND, TOL, SEED + 1);  \
    }

// Default regime, double: the static grid plus the dynamic-only shapes.
TDLS_ANCHOR_CASES(double, 2, 2, "default", 1000, 0.5, 1e-9, 500220)
TDLS_ANCHOR_CASES(double, 7, 4, "default", 1000, 0.5, 1e-9, 500740)
TDLS_ANCHOR_CASES(double, 12, 3, "default", 1000, 0.5, 1e-9, 501230)
TDLS_ANCHOR_CASES(double, 13, 6, "default", 1000, 0.5, 1e-9, 501360)
TDLS_ANCHOR_CASES(double, 25, 5, "default", 400, 0.5, 1e-9, 502550)
TDLS_ANCHOR_CASES(double, 32, 4, "default", 400, 0.5, 1e-9, 503240)
TDLS_ANCHOR_CASES(double, 33, 5, "default", 400, 0.5, 1e-9, 503350)
TDLS_ANCHOR_CASES(double, 64, 8, "default", 120, 0.5, 1e-8, 506480)
TDLS_ANCHOR_CASES(double, 128, 4, "default", 40, 0.5, 1e-8, 512840)
// Tile size exceeding the dimension: a single partial tile.
TDLS_ANCHOR_CASES(double, 5, 8, "default", 1000, 0.5, 1e-9, 500580)
TDLS_ANCHOR_CASES(double, 2, 6, "default", 1000, 0.5, 1e-9, 500260)
// Beyond the static grid.
TDLS_ANCHOR_CASES(double, 100, 3, "default", 60, 0.5, 1e-8, 510030)

// Stress regime, double.
TDLS_ANCHOR_CASES(double, 7, 4, "stress", 1000, 5e-10, 1e-9, 600740)
TDLS_ANCHOR_CASES(double, 12, 3, "stress", 1000, 5e-10, 1e-9, 601230)
TDLS_ANCHOR_CASES(double, 13, 6, "stress", 1000, 5e-10, 1e-9, 601360)
TDLS_ANCHOR_CASES(double, 33, 5, "stress", 400, 5e-10, 1e-9, 603350)
TDLS_ANCHOR_CASES(double, 100, 3, "stress", 60, 5e-10, 1e-8, 610030)

// Float tolerance calibration: the float acceptable-pivot threshold
// (oot_threshold = 1e-4) deliberately keeps in-tile pivots as small as
// 1e-4, so the worst-case element growth is |A|max / 1e-4 and the
// backward error can legitimately reach n * eps_float * growth, around
// 1e-2 on this grid. The reference solver (full partial pivoting) stays
// near eps_float; the gap is the documented cost of the policy, checked
// here, not a defect.
// Float: representative shapes.
TDLS_ANCHOR_CASES(float, 12, 3, "default", 1000, 0.5, 1e-2, 701230)
TDLS_ANCHOR_CASES(float, 5, 8, "default", 1000, 0.5, 1e-2, 700580)

TDLS_TEST_MAIN
