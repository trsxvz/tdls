#ifndef TDLS_TESTS_SOLVERS_RESIDENCY_RUNNER_HPP
#define TDLS_TESTS_SOLVERS_RESIDENCY_RUNNER_HPP



/// \file
/// \brief Shared plumbing of the residency-related suites.
/// \author Tristan Chenaille
///
/// Materializes one system in the storage dictated by the residency
/// template booleans (local arrays for internal components, slices of a
/// small strided arena for external ones), runs one entry path of the
/// static solver, and gathers every output back to contiguous storage so
/// that different residency combinations can be compared bitwise.
///
/// The external arena holds three system slots and the system under test
/// lives in the middle one, so external accesses exercise a stride
/// greater than one and both neighbours can be checked for overwrites by
/// the caller if needed.



#include <cstddef>
#include <vector>

#include <tdls/tdls.hpp>



namespace tdls_tests {



/// \brief Entry paths exercised through the runner.
enum class SolvePath {
    combined, ///< solve()
    split,    ///< factorize() then substitute()
    fused     ///< solve_fused()
};

/// \brief Runs one entry path of the static TiledLUpp solver under one residency
/// combination.
/// \tparam T               scalar type
/// \tparam N               system dimension
/// \tparam TS              tile size
/// \tparam Schedule           elimination schedule
/// \tparam internal_rhs    residency of the right-hand side and solution
/// \tparam internal_piv    residency of the pivot
/// \tparam internal_matrix residency of the matrix
template<typename T, int N, int TS, tdls::TiledLUppSchedule Schedule, bool internal_rhs,
         bool internal_piv, bool internal_matrix>
struct ResidencyRunner {
    static_assert(N <= 32, "the runner keeps internal storage on the stack");

    /// \brief Number of system slots of the external arena.
    static constexpr int arena_count = 3;
    /// \brief Arena slot holding the system under test.
    static constexpr int slot = 1;

    /// \brief Solves one system and returns every output in contiguous
    /// storage.
    /// \param[in]  A0      original matrix, contiguous row-major
    /// \param[in]  b0      right-hand side, contiguous
    /// \param[in]  path    entry path to exercise
    /// \param[out] A_out   factored matrix, contiguous
    /// \param[out] piv_out pivot
    /// \param[out] x_out   solution
    /// \param[out] oot     out-of-tile counter
    /// \return false on a singular matrix.
    static bool run(const T* A0, const T* b0, const SolvePath path, T* A_out, int* piv_out,
                    T* x_out, int& oot) {
        using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Schedule>>;

        // Matrix storage.
        [[maybe_unused]] T A_local[N * N];
        std::vector<T> A_arena;
        T* A;
        int A_stride;
        if constexpr (internal_matrix) {
            for (int e = 0; e < N * N; ++e)
                A_local[e] = A0[e];
            A        = A_local;
            A_stride = 1;
        } else {
            A_arena.assign(static_cast<std::size_t>(N) * N * arena_count, T(0));
            for (int e = 0; e < N * N; ++e)
                A_arena[static_cast<std::size_t>(e) * arena_count + slot] = A0[e];
            A        = A_arena.data() + slot;
            A_stride = arena_count;
        }

        // Pivot storage.
        [[maybe_unused]] int piv_local[N];
        std::vector<int> piv_arena;
        int* piv;
        int piv_stride;
        if constexpr (internal_piv) {
            piv        = piv_local;
            piv_stride = 1;
        } else {
            piv_arena.assign(static_cast<std::size_t>(N) * arena_count, 0);
            piv        = piv_arena.data() + slot;
            piv_stride = arena_count;
        }

        // Right-hand side and solution storage. The fused path solves in
        // place, so the solution slot is preloaded with b there.
        [[maybe_unused]] T b_local[N], x_local[N];
        std::vector<T> b_arena, x_arena;
        const T* b;
        T* x;
        int rhs_stride;
        if constexpr (internal_rhs) {
            for (int i = 0; i < N; ++i) {
                b_local[i] = b0[i];
                x_local[i] = b0[i];
            }
            b          = b_local;
            x          = x_local;
            rhs_stride = 1;
        } else {
            b_arena.assign(static_cast<std::size_t>(N) * arena_count, T(0));
            x_arena.assign(static_cast<std::size_t>(N) * arena_count, T(0));
            for (int i = 0; i < N; ++i) {
                b_arena[static_cast<std::size_t>(i) * arena_count + slot] = b0[i];
                x_arena[static_cast<std::size_t>(i) * arena_count + slot] = b0[i];
            }
            b          = b_arena.data() + slot;
            x          = x_arena.data() + slot;
            rhs_stride = arena_count;
        }

        // Entry path.
        bool ok = false;
        oot     = 0;
        switch (path) {
        case SolvePath::combined:
            ok = Solver::template solve<internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, b, x, rhs_stride, oot);
            break;
        case SolvePath::split:
            ok = Solver::template factorize<internal_piv, internal_matrix>(A, A_stride, piv,
                                                                           piv_stride, oot);
            if (ok)
                Solver::template substitute<internal_rhs, internal_piv, internal_matrix>(
                    A, A_stride, piv, piv_stride, b, x, rhs_stride);
            break;
        case SolvePath::fused:
            ok = Solver::template solve_fused<internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, oot);
            break;
        }

        // Gather every output back to contiguous storage.
        for (int e = 0; e < N * N; ++e) {
            if constexpr (internal_matrix) {
                A_out[e] = A_local[e];
            } else {
                A_out[e] = A_arena[static_cast<std::size_t>(e) * arena_count + slot];
            }
        }
        for (int i = 0; i < N; ++i) {
            if constexpr (internal_piv) {
                piv_out[i] = piv_local[i];
            } else {
                piv_out[i] = piv_arena[static_cast<std::size_t>(i) * arena_count + slot];
            }
            if constexpr (internal_rhs) {
                x_out[i] = x_local[i];
            } else {
                x_out[i] = x_arena[static_cast<std::size_t>(i) * arena_count + slot];
            }
        }
        return ok;
    }
};



} // namespace tdls_tests



#endif // TDLS_TESTS_SOLVERS_RESIDENCY_RUNNER_HPP
