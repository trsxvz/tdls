/// \file
/// \brief Exhaustive cross-product suite on two structurally rich shapes.
/// \author Tristan Chenaille
///
/// The other bridge suites cover each axis against a baseline; this one
/// guards against higher-order interactions by running the FULL cross
/// product of the residency combinations (8), the entry paths (solve,
/// factorize + substitute, solve_fused) and both schedules, on the two
/// most structurally rich shapes: the nominal divisible grid (N = 12,
/// TS = 3) and a trailing-tile grid (N = 13, TS = 6). Every variant must
/// reproduce the fully external combined baseline bitwise.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "residency_runner.hpp"

namespace {

/// \brief Runs one (residency, path) variant on one system and compares
/// it bitwise to the baseline outputs.
/// \tparam T   scalar type
/// \tparam N   system dimension
/// \tparam TS  tile size
/// \tparam Schedule elimination schedule
/// \tparam internal_rhs    residency of the right-hand side under test
/// \tparam internal_piv    residency of the pivot under test
/// \tparam internal_matrix residency of the matrix under test
/// \param[in] A0       original matrix, contiguous
/// \param[in] b0       right-hand side, contiguous
/// \param[in] path     entry path under test
/// \param[in] ok_ref   baseline verdict
/// \param[in] A_ref    baseline factored matrix
/// \param[in] piv_ref  baseline pivot
/// \param[in] x_ref    baseline solution
/// \param[in] oot_ref  baseline out-of-tile counter
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule, bool internal_rhs,
         bool internal_piv, bool internal_matrix>
void check_variant(const T* A0, const T* b0, const tdls_tests::SolvePath path, const bool ok_ref,
                   const T* A_ref, const int* piv_ref, const T* x_ref, const int oot_ref) {
    using Runner = tdls_tests::ResidencyRunner<T, N, TS, Schedule, internal_rhs, internal_piv,
                                               internal_matrix>;
    T A[N * N], x[N];
    int piv[N], oot = 0;
    const bool ok = Runner::run(A0, b0, path, A, piv, x, oot);
    TDLS_CHECK(ok == ok_ref);
    if (!ok || !ok_ref) return;
    TDLS_CHECK_BITWISE(A_ref, A, static_cast<std::size_t>(N) * N);
    TDLS_CHECK_BITWISE(piv_ref, piv, static_cast<std::size_t>(N));
    TDLS_CHECK_BITWISE(x_ref, x, static_cast<std::size_t>(N));
    TDLS_CHECK(oot == oot_ref);
}

/// \brief Runs the full residency x path cross product of one shape and
/// schedule against the fully external combined baseline.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Schedule elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule>
void cross_case(const int count, const double bound, const std::uint64_t seed) {
    using Baseline   = tdls_tests::ResidencyRunner<T, N, TS, Schedule, false, false, false>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    const tdls_tests::SolvePath paths[] = {tdls_tests::SolvePath::combined,
                                           tdls_tests::SolvePath::split,
                                           tdls_tests::SolvePath::fused};
    T A_ref[N * N], x_ref[N];
    int piv_ref[N];
    for (int s = 0; s < count; ++s) {
        int oot_ref = 0;
        const bool ok_ref =
            Baseline::run(batch.matrix(s), batch.rhs(s), tdls_tests::SolvePath::combined, A_ref,
                          piv_ref, x_ref, oot_ref);
        for (const auto path : paths) {
            const T* A0 = batch.matrix(s);
            const T* b0 = batch.rhs(s);
            check_variant<T, N, TS, Schedule, false, false, false>(A0, b0, path, ok_ref, A_ref,
                                                                   piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, false, false, true>(A0, b0, path, ok_ref, A_ref,
                                                                  piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, false, true, false>(A0, b0, path, ok_ref, A_ref,
                                                                  piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, false, true, true>(A0, b0, path, ok_ref, A_ref,
                                                                 piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, true, false, false>(A0, b0, path, ok_ref, A_ref,
                                                                  piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, true, false, true>(A0, b0, path, ok_ref, A_ref,
                                                                 piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, true, true, false>(A0, b0, path, ok_ref, A_ref,
                                                                 piv_ref, x_ref, oot_ref);
            check_variant<T, N, TS, Schedule, true, true, true>(A0, b0, path, ok_ref, A_ref,
                                                                piv_ref, x_ref, oot_ref);
        }
    }
}

} // namespace

TDLS_TEST_CASE("tiledlupp/cross-full/double/N=12,TS=3,RL,default") {
    cross_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(60, 0.5, 220100);
}
TDLS_TEST_CASE("tiledlupp/cross-full/double/N=12,TS=3,RL,stress") {
    cross_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(60, 5e-10, 220200);
}
TDLS_TEST_CASE("tiledlupp/cross-full/double/N=12,TS=3,LL,default") {
    cross_case<double, 12, 3, tdls::TiledLUppSchedule::LeftLooking>(60, 0.5, 220300);
}
TDLS_TEST_CASE("tiledlupp/cross-full/double/N=13,TS=6,RL,default") {
    cross_case<double, 13, 6, tdls::TiledLUppSchedule::RightLooking>(60, 0.5, 220400);
}
TDLS_TEST_CASE("tiledlupp/cross-full/double/N=13,TS=6,LL,default") {
    cross_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(60, 0.5, 220500);
}
TDLS_TEST_CASE("tiledlupp/cross-full/double/N=13,TS=6,LL,stress") {
    cross_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(60, 5e-10, 220600);
}

TDLS_TEST_MAIN
