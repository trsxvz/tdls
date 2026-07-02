/// \file
/// \brief Anchor suite of the static TiledLUpp solver.
/// \author Tristan Chenaille
///
/// TiledLUppSolverStatic is compared against the independent reference LU
/// (tests/common/reference_lu.hpp) on a grid of shapes covering the
/// structural boundaries of the algorithm: minimal N, trailing tiles,
/// the 32/33 mask boundary of substitute_inplace, TS = N, and large N
/// exercising the cycle-leader path. Both schedules, both scalar types
/// and both input regimes (default +-0.5 and out-of-tile stress +-5e-10)
/// are covered. The verdict of every case is the normwise backward error
/// of both solvers, plus the exact agreement of the singularity verdicts
/// (one structurally singular system is injected in every batch).

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Runs one anchor comparison on a reproducible batch: solves with
/// the tiled solver (contiguous external storage, pivot in a local array)
/// and with the reference, requires identical verdicts, exactly one
/// singular report on each side, and both backward errors under the
/// tolerance.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Sched elimination schedule
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched>
void anchor_case(const int count, const double bound, const double tolerance,
                 const std::uint64_t seed) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Sched>>;
    auto batch   = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree    = true;
    int singular_tiled     = 0;
    int singular_reference = 0;
    double be_tiled        = 0.0;
    double be_reference    = 0.0;

    std::vector<T> At(N * N), Ar(N * N), x(N), xr(N);
    int piv[N], pivr[N];
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, At.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, Ar.begin());
        const bool ok_tiled = Solver::template solve<false, true, false>(At.data(), 1, piv, 1,
                                                                         batch.rhs(s), x.data(), 1);
        const bool ok_reference =
            tdls_tests::reference_solve(Ar.data(), pivr, batch.rhs(s), xr.data(), N);
        if (ok_tiled != ok_reference) verdicts_agree = false;
        if (!ok_tiled) ++singular_tiled;
        if (!ok_reference) ++singular_reference;
        if (ok_tiled && ok_reference) {
            be_tiled = std::max(
                be_tiled, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), N));
            be_reference =
                std::max(be_reference,
                         tdls_tests::backward_error(batch.matrix(s), xr.data(), batch.rhs(s), N));
        }
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(singular_tiled == 1);
    TDLS_CHECK(singular_reference == 1);
    TDLS_CHECK_LE(be_tiled, tolerance);
    TDLS_CHECK_LE(be_reference, tolerance);
}

} // namespace

/// Emits the RL and LL anchor cases of one (type, N, TS, regime) cell.
#define TDLS_ANCHOR_CASES(T, N, TS, REGIME, COUNT, BOUND, TOL, SEED)                               \
    TDLS_TEST_CASE("tiledlupp/oracle/static/" #T "/N=" #N ",TS=" #TS ",RL," REGIME) {              \
        anchor_case<T, N, TS, tdls::TiledLUppSchedule::RightLooking>(COUNT, BOUND, TOL, SEED);     \
    }                                                                                              \
    TDLS_TEST_CASE("tiledlupp/oracle/static/" #T "/N=" #N ",TS=" #TS ",LL," REGIME) {              \
        anchor_case<T, N, TS, tdls::TiledLUppSchedule::LeftLooking>(COUNT, BOUND, TOL, SEED + 1);  \
    }

// Default regime, double: the whole shape grid.
TDLS_ANCHOR_CASES(double, 2, 2, "default", 1000, 0.5, 1e-9, 100220)
TDLS_ANCHOR_CASES(double, 7, 4, "default", 1000, 0.5, 1e-9, 100740)
TDLS_ANCHOR_CASES(double, 12, 3, "default", 1000, 0.5, 1e-9, 101230)
TDLS_ANCHOR_CASES(double, 13, 6, "default", 1000, 0.5, 1e-9, 101360)
TDLS_ANCHOR_CASES(double, 25, 5, "default", 400, 0.5, 1e-9, 102550)
TDLS_ANCHOR_CASES(double, 32, 4, "default", 400, 0.5, 1e-9, 103240)
TDLS_ANCHOR_CASES(double, 33, 5, "default", 400, 0.5, 1e-9, 103350)
TDLS_ANCHOR_CASES(double, 64, 8, "default", 120, 0.5, 1e-8, 106480)
TDLS_ANCHOR_CASES(double, 128, 4, "default", 40, 0.5, 1e-8, 112840)

// Stress regime, double: tiny entries put every in-tile pivot below
// oot_threshold, so the out-of-tile recovery fires on most columns.
TDLS_ANCHOR_CASES(double, 2, 2, "stress", 1000, 5e-10, 1e-9, 200220)
TDLS_ANCHOR_CASES(double, 7, 4, "stress", 1000, 5e-10, 1e-9, 200740)
TDLS_ANCHOR_CASES(double, 12, 3, "stress", 1000, 5e-10, 1e-9, 201230)
TDLS_ANCHOR_CASES(double, 13, 6, "stress", 1000, 5e-10, 1e-9, 201360)
TDLS_ANCHOR_CASES(double, 25, 5, "stress", 400, 5e-10, 1e-9, 202550)
TDLS_ANCHOR_CASES(double, 32, 4, "stress", 400, 5e-10, 1e-9, 203240)
TDLS_ANCHOR_CASES(double, 33, 5, "stress", 400, 5e-10, 1e-9, 203350)
TDLS_ANCHOR_CASES(double, 64, 8, "stress", 120, 5e-10, 1e-8, 206480)
TDLS_ANCHOR_CASES(double, 128, 4, "stress", 40, 5e-10, 1e-8, 212840)

// Float tolerance calibration: the float acceptable-pivot threshold
// (oot_threshold = 1e-4) deliberately keeps in-tile pivots as small as
// 1e-4, so the worst-case element growth is |A|max / 1e-4 and the
// backward error can legitimately reach n * eps_float * growth, around
// 1e-2 on this grid. The reference solver (full partial pivoting) stays
// near eps_float; the gap is the documented cost of the policy, checked
// here, not a defect.
// Float: representative shapes (nominal and trailing tile). The float
// oot_threshold (1e-4) makes the stress regime fire the out-of-tile
// search on every column.
TDLS_ANCHOR_CASES(float, 12, 3, "default", 1000, 0.5, 1e-2, 301230)
TDLS_ANCHOR_CASES(float, 13, 6, "default", 1000, 0.5, 1e-2, 301360)
TDLS_ANCHOR_CASES(float, 12, 3, "stress", 1000, 5e-10, 1e-2, 401230)

TDLS_TEST_MAIN
