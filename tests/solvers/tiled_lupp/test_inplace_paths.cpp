/// \file
/// \brief Bridge suite: the in-place substitution paths are equivalent.
/// \author Tristan Chenaille
///
/// substitute_inplace dispatches between three mechanisms depending on
/// the dimension: a 32-bit visited mask (static, N <= 32), a 64-bit mask
/// (static 33..64, dynamic n <= 64) and a cycle-leader scan (beyond).
/// All of them apply the same permutation, so on identical inputs the
/// in-place result must be bitwise identical to the gather-based
/// substitute(), and the static and dynamic in-place results must be
/// bitwise identical to each other. The dimensions bracket every
/// mechanism boundary: 31/32/33 and 63/64/65, plus 100 deep in the
/// cycle-leader domain.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief On one reproducible batch: factors each system once, then
/// checks substitute_inplace against substitute on the same
/// factorization, and the dynamic in-place path against the static one.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS>
void inplace_case(const int count, const double bound, const std::uint64_t seed) {
    using Config     = tdls::TiledLUppConfig<T, TS>;
    using Static     = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic    = tdls::TiledLUppSolverDynamic<T, Config>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> A_static(static_cast<std::size_t>(N) * N);
    std::vector<T> A_dynamic(static_cast<std::size_t>(N) * N);
    std::vector<T> x_gather(N), x_static(N), x_dynamic(N);
    std::vector<int> piv_static(N), piv_dynamic(N);
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_static.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_dynamic.begin());
        const bool ok_static =
            Static::template factorize<false, false>(A_static.data(), 1, piv_static.data(), 1);
        const bool ok_dynamic = Dynamic::factorize(N, A_dynamic.data(), 1, piv_dynamic.data(), 1);
        TDLS_CHECK(ok_static == ok_dynamic);
        if (!ok_static || !ok_dynamic) continue;

        Static::template substitute<false, false, false>(A_static.data(), 1, piv_static.data(), 1,
                                                         batch.rhs(s), x_gather.data(), 1);
        std::copy(batch.rhs(s), batch.rhs(s) + N, x_static.begin());
        Static::template substitute_inplace<false, false, false>(
            A_static.data(), 1, piv_static.data(), 1, x_static.data(), 1);
        std::copy(batch.rhs(s), batch.rhs(s) + N, x_dynamic.begin());
        Dynamic::substitute_inplace(N, A_dynamic.data(), 1, piv_dynamic.data(), 1, x_dynamic.data(),
                                    1);

        TDLS_CHECK_BITWISE(x_gather.data(), x_static.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x_static.data(), x_dynamic.data(), static_cast<std::size_t>(N));
    }
}

} // namespace

/// Emits the in-place bridge case of one (N, TS, regime) cell.
#define TDLS_INPLACE_CASE(N, TS, REGIME, COUNT, BOUND, SEED)                                       \
    TDLS_TEST_CASE("tiledlupp/bridge/inplace/double/N=" #N ",TS=" #TS "," REGIME) {                \
        inplace_case<double, N, TS>(COUNT, BOUND, SEED);                                           \
    }

// The 32-bit mask domain and its upper boundary.
TDLS_INPLACE_CASE(31, 4, "default", 150, 0.5, 160310)
TDLS_INPLACE_CASE(32, 4, "default", 150, 0.5, 160320)
// The 64-bit mask domain and its two boundaries.
TDLS_INPLACE_CASE(33, 5, "default", 150, 0.5, 160330)
TDLS_INPLACE_CASE(33, 5, "stress", 150, 5e-10, 160331)
TDLS_INPLACE_CASE(63, 7, "default", 100, 0.5, 160630)
TDLS_INPLACE_CASE(64, 8, "default", 100, 0.5, 160640)
TDLS_INPLACE_CASE(64, 8, "stress", 100, 5e-10, 160641)
// The cycle-leader domain.
TDLS_INPLACE_CASE(65, 5, "default", 100, 0.5, 160650)
TDLS_INPLACE_CASE(100, 5, "default", 60, 0.5, 161000)

TDLS_TEST_MAIN
