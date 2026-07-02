/// \file
/// \brief Suite of the TiledLUppSolverConfig knobs.
/// \author Tristan Chenaille
///
/// Each compile-time knob is checked through its observable contract:
/// unroll_inner never changes any value (bitwise equivalence of both
/// settings); oot_first_acceptable changes which acceptable pivot wins
/// but keeps both settings anchored on the backward error;
/// the out-of-tile counter stays at zero in the default regime, fires in
/// the stress regime, and counts every column when the threshold is
/// raised above any entry; a raised singular_eps turns tiny-pivot
/// batches into singular verdicts.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Configuration flipping only unroll_inner.
template<typename T, int TS, tdls::TiledLUppSchedule Sched>
struct NoUnrollConfig : tdls::TiledLUppConfig<T, TS, Sched> {
    static constexpr bool unroll_inner = false;
};

/// \brief Configuration restoring the full-panel out-of-tile scan.
template<typename T, int TS>
struct MaxScanConfig : tdls::TiledLUppConfig<T, TS> {
    static constexpr bool oot_first_acceptable = false;
};

/// \brief Configuration whose threshold no entry can reach: the
/// out-of-tile search fires on every column.
template<typename T, int TS>
struct AlwaysOotConfig : tdls::TiledLUppConfig<T, TS> {
    static constexpr T oot_threshold = T(1e30);
};

/// \brief Configuration declaring every pivot below 1e-3 singular. The
/// floor only guards the out-of-tile recovery path, so the acceptance
/// threshold is raised together with it (the TiledLUpp solvers enforce
/// singular_eps <= oot_threshold at compile time).
template<typename T, int TS>
struct StrictEpsConfig : tdls::TiledLUppConfig<T, TS> {
    static constexpr T oot_threshold = T(1e-3);
    static constexpr T singular_eps  = T(1e-3);
};

/// \brief Checks that unroll_inner = false reproduces the default
/// bitwise: the knob moves pragmas, never values.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Sched elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched>
void unroll_case(const int count, const double bound, const std::uint64_t seed) {
    using Default    = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Sched>>;
    using NoUnroll   = tdls::TiledLUppSolverStatic<T, N, NoUnrollConfig<T, TS, Sched>>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> A_def(N * N), A_alt(N * N), x_def(N), x_alt(N);
    int piv_def[N], piv_alt[N];
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_def.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_alt.begin());
        const bool ok_def = Default::template solve<false, true, false>(
            A_def.data(), 1, piv_def, 1, batch.rhs(s), x_def.data(), 1);
        const bool ok_alt = NoUnroll::template solve<false, true, false>(
            A_alt.data(), 1, piv_alt, 1, batch.rhs(s), x_alt.data(), 1);
        TDLS_CHECK(ok_def == ok_alt);
        if (!ok_def) continue;
        TDLS_CHECK_BITWISE(A_def.data(), A_alt.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv_def, piv_alt, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x_def.data(), x_alt.data(), static_cast<std::size_t>(N));
    }
}

} // namespace

TDLS_TEST_CASE("tiledlupp/config/unroll_inner-off-is-bitwise/N=12,TS=3,RL,default") {
    unroll_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 190100);
}
TDLS_TEST_CASE("tiledlupp/config/unroll_inner-off-is-bitwise/N=13,TS=6,LL,stress") {
    unroll_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(200, 5e-10, 190200);
}

TDLS_TEST_CASE("tiledlupp/config/oot_first_acceptable-both-anchored/N=25,TS=5,stress") {
    // The two search strategies may pick different acceptable pivots, so
    // the results differ; both must stay anchored on the backward error.
    constexpr int N  = 25;
    using FirstOk    = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 5>>;
    using MaxScan    = tdls::TiledLUppSolverStatic<double, N, MaxScanConfig<double, 5>>;
    const auto batch = tdls_tests::make_batch<double>(N, 300, 190300, 5e-10);
    std::vector<double> A(N * N), x(N);
    std::vector<int> piv(N);
    double be_first = 0.0, be_max_scan = 0.0;
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        if (FirstOk::solve<false, false, false>(A.data(), 1, piv.data(), 1, batch.rhs(s), x.data(),
                                                1))
            be_first = std::max(
                be_first, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), N));
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        if (MaxScan::solve<false, false, false>(A.data(), 1, piv.data(), 1, batch.rhs(s), x.data(),
                                                1))
            be_max_scan =
                std::max(be_max_scan,
                         tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), N));
    }
    TDLS_CHECK_LE(be_first, 1e-9);
    TDLS_CHECK_LE(be_max_scan, 1e-9);
}

TDLS_TEST_CASE("tiledlupp/config/oot-counter/silent-in-default-regime/N=12,TS=3") {
    constexpr int N  = 12;
    using Solver     = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;
    const auto batch = tdls_tests::make_batch<double>(N, 200, 190400, 0.5);
    std::vector<double> A(N * N), x(N);
    int piv[N];
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        int oot = -1;
        if (Solver::solve<false, true, false>(A.data(), 1, piv, 1, batch.rhs(s), x.data(), 1, oot))
            TDLS_CHECK(oot == 0);
    }
}

TDLS_TEST_CASE("tiledlupp/config/oot-counter/fires-in-stress-regime/N=12,TS=3") {
    constexpr int N  = 12;
    using Solver     = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;
    const auto batch = tdls_tests::make_batch<double>(N, 200, 190500, 5e-10);
    std::vector<double> A(N * N), x(N);
    int piv[N];
    long total = 0;
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        int oot = 0;
        if (Solver::solve<false, true, false>(A.data(), 1, piv, 1, batch.rhs(s), x.data(), 1, oot))
            total += oot;
    }
    TDLS_CHECK(total > 0);
}

TDLS_TEST_CASE("tiledlupp/config/oot-counter/every-column-when-threshold-unreachable/N=12,TS=3") {
    constexpr int N  = 12;
    using Solver     = tdls::TiledLUppSolverStatic<double, N, AlwaysOotConfig<double, 3>>;
    const auto batch = tdls_tests::make_batch<double>(N, 100, 190600, 0.5);
    std::vector<double> A(N * N), x(N);
    int piv[N];
    double be_max = 0.0;
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        int oot = 0;
        if (Solver::solve<false, true, false>(A.data(), 1, piv, 1, batch.rhs(s), x.data(), 1,
                                              oot)) {
            TDLS_CHECK(oot == N);
            be_max = std::max(
                be_max, tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), N));
        }
    }
    TDLS_CHECK_LE(be_max, 1e-9);
}

TDLS_TEST_CASE("tiledlupp/config/strict-singular_eps-rejects-tiny-pivots/N=12,TS=3") {
    // Stress entries never exceed 5e-10: with singular_eps = 1e-3 every
    // system is declared singular, while the default floor solves them.
    constexpr int N  = 12;
    using Strict     = tdls::TiledLUppSolverStatic<double, N, StrictEpsConfig<double, 3>>;
    using Default    = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;
    const auto batch = tdls_tests::make_batch<double>(N, 100, 190700, 5e-10);
    std::vector<double> A(N * N), x(N);
    int piv[N];
    for (int s = 0; s < batch.count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        TDLS_CHECK(
            (!Strict::solve<false, true, false>(A.data(), 1, piv, 1, batch.rhs(s), x.data(), 1)));
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A.begin());
        TDLS_CHECK(
            (Default::solve<false, true, false>(A.data(), 1, piv, 1, batch.rhs(s), x.data(), 1)));
    }
}

TDLS_TEST_MAIN
