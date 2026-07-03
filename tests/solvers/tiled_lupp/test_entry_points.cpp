/// \file
/// \brief Bridge suite: the entry points are mutually consistent.
/// \author Tristan Chenaille
///
/// The equivalences documented by the TiledLUpp solvers are checked bitwise on
/// identical inputs: solve() against factorize() + substitute(),
/// solve_inplace() against factorize() + substitute_inplace(),
/// substitute_canonical() against substitute_canonical_block<1>(), a
/// block of canonical columns against the same columns solved one by
/// one, the reuse of one factorization across several right-hand sides,
/// and the diagnostics-free overloads against the counting ones.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Runs every entry-point equivalence on one reproducible batch.
/// The matrix is external contiguous, pivot and right-hand sides are
/// internal; the residency bridge covers the other combinations.
/// \tparam T     scalar type
/// \tparam N     system dimension
/// \tparam TS    tile size
/// \tparam Schedule elimination schedule
/// \param[in] count number of systems
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule>
void entry_points_case(const int count, const double bound, const std::uint64_t seed) {
    using Solver     = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Schedule>>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    std::vector<T> A_split(N * N), A_other(N * N);
    for (int s = 0; s < count; ++s) {
        const T* A0 = batch.matrix(s);
        const T* b0 = batch.rhs(s);

        // Baseline: factorize then substitute.
        std::copy(A0, A0 + N * N, A_split.begin());
        int piv_split[N], oot_split = 0;
        T b[N], x_split[N];
        for (int i = 0; i < N; ++i)
            b[i] = b0[i];
        const bool ok =
            Solver::template factorize<true, false>(A_split.data(), 1, piv_split, 1, oot_split);
        if (!ok) continue;
        Solver::template substitute<true, true, false>(A_split.data(), 1, piv_split, 1, b, x_split,
                                                       1);

        // solve() must reproduce the split path.
        {
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N], oot = 0;
            T x[N];
            const bool ok_solve =
                Solver::template solve<true, true, false>(A_other.data(), 1, piv, 1, b, x, 1, oot);
            TDLS_CHECK(ok_solve);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_split, x, static_cast<std::size_t>(N));
            TDLS_CHECK(oot_split == oot);
        }

        // solve_inplace() must reproduce factorize + substitute_inplace,
        // which itself must reproduce the separate substitution.
        {
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N], oot = 0;
            T y[N];
            for (int i = 0; i < N; ++i)
                y[i] = b0[i];
            const bool ok_fused = Solver::template solve_inplace<true, true, false>(
                A_other.data(), 1, piv, 1, y, 1, oot);
            TDLS_CHECK(ok_fused);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv, static_cast<std::size_t>(N));
            TDLS_CHECK_BITWISE(x_split, y, static_cast<std::size_t>(N));
            TDLS_CHECK(oot_split == oot);

            T x_inplace[N];
            for (int i = 0; i < N; ++i)
                x_inplace[i] = b0[i];
            Solver::template substitute_inplace<true, true, false>(A_split.data(), 1, piv_split, 1,
                                                                   x_inplace, 1);
            TDLS_CHECK_BITWISE(x_split, x_inplace, static_cast<std::size_t>(N));
        }

        // substitute_canonical must alias substitute_canonical_block<1>,
        // and a block of columns must match the same columns solved one
        // by one.
        {
            T x_one[N], x_alias[N], x_block[3 * N];
            for (int c = 0; c < N; ++c) {
                Solver::template substitute_canonical<true, true, false>(A_split.data(), 1,
                                                                         piv_split, 1, c, x_one, 1);
                Solver::template substitute_canonical_block<1, true, true, false>(
                    A_split.data(), 1, piv_split, 1, c, x_alias, 1, 0);
                TDLS_CHECK_BITWISE(x_one, x_alias, static_cast<std::size_t>(N));
            }
            Solver::template substitute_canonical_block<3, true, true, false>(
                A_split.data(), 1, piv_split, 1, 0, x_block, 1, 0);
            for (int c = 0; c < 3; ++c) {
                Solver::template substitute_canonical<true, true, false>(A_split.data(), 1,
                                                                         piv_split, 1, c, x_one, 1);
                TDLS_CHECK_BITWISE(x_one, x_block + c * N, static_cast<std::size_t>(N));
            }
        }

        // One factorization reused across several right-hand sides must
        // match fresh solves of the same systems.
        {
            const int other = (s + 1) % count;
            T b2[N], x_reuse[N], x_fresh[N];
            for (int i = 0; i < N; ++i)
                b2[i] = batch.rhs(other)[i];
            Solver::template substitute<true, true, false>(A_split.data(), 1, piv_split, 1, b2,
                                                           x_reuse, 1);
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N];
            if (Solver::template solve<true, true, false>(A_other.data(), 1, piv, 1, b2, x_fresh,
                                                          1))
                TDLS_CHECK_BITWISE(x_reuse, x_fresh, static_cast<std::size_t>(N));
        }

        // The diagnostics-free factorize overload must produce the same
        // factorization as the counting one.
        {
            std::copy(A0, A0 + N * N, A_other.begin());
            int piv[N];
            const bool ok_free = Solver::template factorize<true, false>(A_other.data(), 1, piv, 1);
            TDLS_CHECK(ok_free);
            TDLS_CHECK_BITWISE(A_split.data(), A_other.data(), static_cast<std::size_t>(N) * N);
            TDLS_CHECK_BITWISE(piv_split, piv, static_cast<std::size_t>(N));
        }
    }
}

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/entry-points/double/N=12,TS=3,RL,default") {
    entry_points_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 150100);
}
TDLS_TEST_CASE("tiledlupp/bridge/entry-points/double/N=12,TS=3,RL,stress") {
    entry_points_case<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 5e-10, 150200);
}
TDLS_TEST_CASE("tiledlupp/bridge/entry-points/double/N=13,TS=6,LL,default") {
    entry_points_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(200, 0.5, 150300);
}
TDLS_TEST_CASE("tiledlupp/bridge/entry-points/double/N=13,TS=6,LL,stress") {
    entry_points_case<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(200, 5e-10, 150400);
}
TDLS_TEST_CASE("tiledlupp/bridge/entry-points/float/N=12,TS=3,RL,default") {
    entry_points_case<float, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 150500);
}

TDLS_TEST_MAIN
