/// \file
/// \brief Bridge suite: the eight residency combinations are equivalent.
/// \author Tristan Chenaille
///
/// The internal_rhs / internal_piv / internal_matrix template booleans
/// select where each component lives (caller-local array or strided
/// remote storage) but never change the arithmetic: on identical inputs,
/// all eight combinations must produce bitwise-identical factored
/// matrices, pivots, solutions and out-of-tile counters. The baseline is
/// the fully external combination; the other seven are compared to it on
/// shapes covering trailing tiles and both schedules.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "residency_runner.hpp"

namespace {

/// \brief Compares one residency combination to the fully external
/// baseline over a reproducible batch, on the combined solve path.
/// \tparam T   scalar type
/// \tparam N   system dimension
/// \tparam TS  tile size
/// \tparam Sched elimination schedule
/// \tparam internal_rhs    residency of the right-hand side under test
/// \tparam internal_piv    residency of the pivot under test
/// \tparam internal_matrix residency of the matrix under test
/// \param[in] batch input systems
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched, bool internal_rhs,
         bool internal_piv, bool internal_matrix>
void compare_to_baseline(const tdls_tests::SystemBatch<T>& batch) {
    using Baseline = tdls_tests::ResidencyRunner<T, N, TS, Sched, false, false, false>;
    using Tested =
        tdls_tests::ResidencyRunner<T, N, TS, Sched, internal_rhs, internal_piv, internal_matrix>;

    std::vector<T> A_ref(N * N), A_tst(N * N), x_ref(N), x_tst(N);
    int piv_ref[N], piv_tst[N];
    for (int s = 0; s < batch.count; ++s) {
        int oot_ref = 0, oot_tst = 0;
        const bool ok_ref =
            Baseline::run(batch.matrix(s), batch.rhs(s), tdls_tests::SolvePath::combined,
                          A_ref.data(), piv_ref, x_ref.data(), oot_ref);
        const bool ok_tst =
            Tested::run(batch.matrix(s), batch.rhs(s), tdls_tests::SolvePath::combined,
                        A_tst.data(), piv_tst, x_tst.data(), oot_tst);
        TDLS_CHECK(ok_ref == ok_tst);
        if (!ok_ref || !ok_tst) continue;
        TDLS_CHECK_BITWISE(A_ref.data(), A_tst.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv_ref, piv_tst, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x_ref.data(), x_tst.data(), static_cast<std::size_t>(N));
        TDLS_CHECK(oot_ref == oot_tst);
    }
}

/// \brief Runs the seven non-baseline combinations against the baseline.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Sched elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched>
void all_combos_case(const int count, const double bound, const std::uint64_t seed) {
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);
    compare_to_baseline<T, N, TS, Sched, false, false, true>(batch);
    compare_to_baseline<T, N, TS, Sched, false, true, false>(batch);
    compare_to_baseline<T, N, TS, Sched, false, true, true>(batch);
    compare_to_baseline<T, N, TS, Sched, true, false, false>(batch);
    compare_to_baseline<T, N, TS, Sched, true, false, true>(batch);
    compare_to_baseline<T, N, TS, Sched, true, true, false>(batch);
    compare_to_baseline<T, N, TS, Sched, true, true, true>(batch);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=12,TS=3,RL,default") {
    all_combos_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 130100);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=12,TS=3,RL,stress") {
    all_combos_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 5e-10, 130200);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=12,TS=3,LL,stress") {
    all_combos_case<double, 12, 3, tdls::TiledLUppSchedule::LeftLooking>(200, 5e-10, 130300);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=13,TS=6,RL,default") {
    all_combos_case<double, 13, 6, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 130400);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=13,TS=6,LL,default") {
    all_combos_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(200, 0.5, 130500);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/double/N=7,TS=4,RL,default") {
    all_combos_case<double, 7, 4, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 130600);
}
TDLS_TEST_CASE("tiledlupp/bridge/residencies/float/N=12,TS=3,RL,default") {
    all_combos_case<float, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 130700);
}

TDLS_TEST_MAIN
