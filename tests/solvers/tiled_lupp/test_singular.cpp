/// \file
/// \brief Suite of the singularity verdicts.
/// \author Tristan Chenaille
///
/// The solvers declare a matrix singular only when the best available
/// pivot is zero or subnormal (below numeric_limits::min()), mirroring
/// the reference solver. On a set of structurally singular constructions
/// (zero column at several positions, zero row, all-zero matrix, fully
/// subnormal matrix) and on solvable extreme cases (uniformly tiny but
/// normal entries), the verdicts of the static TiledLUpp solver (both schedules),
/// the dynamic TiledLUpp solver and the reference must all agree.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

constexpr int N  = 12;
constexpr int TS = 3;

/// \brief Runs the four solvers on one contiguous system and checks that
/// every verdict matches the expectation.
/// \param[in] A0       matrix, contiguous row-major
/// \param[in] b0       right-hand side
/// \param[in] solvable expected verdict
void check_verdicts(const double* A0, const double* b0, const bool solvable) {
    using ConfigRL = tdls::TiledLUppConfig<double, TS, tdls::TiledLUppSchedule::RightLooking>;
    using ConfigLL = tdls::TiledLUppConfig<double, TS, tdls::TiledLUppSchedule::LeftLooking>;
    std::vector<double> A(N * N), x(N);
    std::vector<int> piv(N);

    std::copy(A0, A0 + N * N, A.begin());
    const bool ok_rl = tdls::TiledLUppSolverStatic<double, N, ConfigRL>::solve<false, false, false>(
        A.data(), 1, piv.data(), 1, b0, x.data(), 1);
    TDLS_CHECK(ok_rl == solvable);

    std::copy(A0, A0 + N * N, A.begin());
    const bool ok_ll = tdls::TiledLUppSolverStatic<double, N, ConfigLL>::solve<false, false, false>(
        A.data(), 1, piv.data(), 1, b0, x.data(), 1);
    TDLS_CHECK(ok_ll == solvable);

    std::copy(A0, A0 + N * N, A.begin());
    const bool ok_dyn = tdls::TiledLUppSolverDynamic<double, ConfigRL>::solve(
        N, A.data(), 1, piv.data(), 1, b0, x.data(), 1);
    TDLS_CHECK(ok_dyn == solvable);

    std::copy(A0, A0 + N * N, A.begin());
    std::vector<double> xr(N);
    const bool ok_ref = tdls_tests::reference_solve(A.data(), piv.data(), b0, xr.data(), N);
    TDLS_CHECK(ok_ref == solvable);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/singular/zero-column-first") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 180100, 0.5);
    tdls_tests::zero_column(batch, 0, 0);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/zero-column-middle") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 180200, 0.5);
    tdls_tests::zero_column(batch, 0, 5);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/zero-column-last") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 180300, 0.5);
    tdls_tests::zero_column(batch, 0, N - 1);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/zero-row") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 180400, 0.5);
    for (int j = 0; j < N; ++j)
        batch.matrix(0)[3 * N + j] = 0.0;
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/all-zero-matrix") {
    auto batch = tdls_tests::make_batch<double>(N, 1, 180500, 0.5);
    std::fill(batch.matrix(0), batch.matrix(0) + N * N, 0.0);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/subnormal-matrix") {
    // Every entry is subnormal, so every pivot candidate is below the
    // singularity floor.
    auto batch = tdls_tests::make_batch<double>(N, 1, 180600, 1e-310);
    check_verdicts(batch.matrix(0), batch.rhs(0), false);
}

TDLS_TEST_CASE("tiledlupp/singular/tiny-but-normal-matrix-is-solvable") {
    // Uniformly tiny normal entries stay well above the floor: a small
    // scale is not a singularity (this is the rationale of the
    // numeric_limits::min() criterion).
    auto batch = tdls_tests::make_batch<double>(N, 1, 180700, 1e-30);
    check_verdicts(batch.matrix(0), batch.rhs(0), true);
}

TDLS_TEST_MAIN
