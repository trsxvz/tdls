/// \file
/// \brief Bridge suite: the dynamic TiledLUpp solver reproduces the static TiledLUpp solver
/// bitwise.
/// \author Tristan Chenaille
///
/// At equal shape (N, TS, schedule, configuration) and on identical
/// inputs, TiledLUppSolverDynamic and TiledLUppSolverStatic execute the same
/// arithmetic sequence: the factored matrix, the pivot, the solution and
/// the out-of-tile counter must be bitwise identical. The bridge runs on
/// the boundary-covering shape grid so that every structural code path
/// (trailing tiles, mask widths, cycle-leader, TS = N) is crossed by the
/// equivalence proof.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Solves every system of a reproducible batch with both solvers
/// on contiguous external storage and checks the bitwise equality of all
/// outputs, including the singularity verdicts and the out-of-tile
/// counters.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Sched elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched>
void bridge_case(const int count, const double bound, const std::uint64_t seed) {
    using Config  = tdls::TiledLUppConfig<T, TS, Sched>;
    using Static  = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic = tdls::TiledLUppSolverDynamic<T, Config>;
    auto batch    = tdls_tests::make_batch<T>(N, count, seed, bound);
    tdls_tests::zero_column(batch, 0, 0);

    bool verdicts_agree = true;
    int matched         = 0;

    std::vector<T> As(N * N), Ad(N * N), xs(N), xd(N);
    int pivs[N], pivd[N];
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, As.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, Ad.begin());
        int oot_static       = 0;
        int oot_dynamic      = 0;
        const bool ok_static = Static::template solve<false, true, false>(
            As.data(), 1, pivs, 1, batch.rhs(s), xs.data(), 1, oot_static);
        const bool ok_dynamic =
            Dynamic::solve(N, Ad.data(), 1, pivd, 1, batch.rhs(s), xd.data(), 1, oot_dynamic);
        if (ok_static != ok_dynamic) verdicts_agree = false;
        if (!ok_static || !ok_dynamic) continue;
        ++matched;
        TDLS_CHECK_BITWISE(As.data(), Ad.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(xs.data(), xd.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(pivs, pivd, static_cast<std::size_t>(N));
        TDLS_CHECK(oot_static == oot_dynamic);
    }
    TDLS_CHECK(verdicts_agree);
    TDLS_CHECK(matched == count - 1); // every system but the singular one
}

} // namespace

/// Emits the RL and LL bridge cases of one (type, N, TS, regime) cell.
#define TDLS_BRIDGE_CASES(T, N, TS, REGIME, COUNT, BOUND, SEED)                                    \
    TDLS_TEST_CASE("tiledlupp/bridge/static-dynamic/" #T "/N=" #N ",TS=" #TS ",RL," REGIME) {      \
        bridge_case<T, N, TS, tdls::TiledLUppSchedule::RightLooking>(COUNT, BOUND, SEED);          \
    }                                                                                              \
    TDLS_TEST_CASE("tiledlupp/bridge/static-dynamic/" #T "/N=" #N ",TS=" #TS ",LL," REGIME) {      \
        bridge_case<T, N, TS, tdls::TiledLUppSchedule::LeftLooking>(COUNT, BOUND, SEED + 1);       \
    }

// Default regime over the boundary-covering grid, double.
TDLS_BRIDGE_CASES(double, 2, 2, "default", 400, 0.5, 810220)
TDLS_BRIDGE_CASES(double, 7, 4, "default", 400, 0.5, 810740)
TDLS_BRIDGE_CASES(double, 12, 3, "default", 400, 0.5, 811230)
TDLS_BRIDGE_CASES(double, 12, 12, "default", 400, 0.5, 811212)
TDLS_BRIDGE_CASES(double, 13, 6, "default", 400, 0.5, 811360)
TDLS_BRIDGE_CASES(double, 25, 5, "default", 200, 0.5, 812550)
TDLS_BRIDGE_CASES(double, 32, 4, "default", 200, 0.5, 813240)
TDLS_BRIDGE_CASES(double, 33, 5, "default", 200, 0.5, 813350)
TDLS_BRIDGE_CASES(double, 64, 8, "default", 60, 0.5, 816480)
TDLS_BRIDGE_CASES(double, 128, 4, "default", 30, 0.5, 822840)

// Stress regime: the out-of-tile recovery and its counter are exercised
// on most columns, so the counter equality check carries real weight.
TDLS_BRIDGE_CASES(double, 12, 3, "stress", 400, 5e-10, 911230)
TDLS_BRIDGE_CASES(double, 13, 6, "stress", 400, 5e-10, 911360)
TDLS_BRIDGE_CASES(double, 33, 5, "stress", 200, 5e-10, 913350)
TDLS_BRIDGE_CASES(double, 128, 4, "stress", 30, 5e-10, 922840)

// Float: one nominal and one trailing-tile shape.
TDLS_BRIDGE_CASES(float, 12, 3, "default", 400, 0.5, 951230)
TDLS_BRIDGE_CASES(float, 13, 6, "default", 400, 0.5, 951360)

TDLS_TEST_MAIN
