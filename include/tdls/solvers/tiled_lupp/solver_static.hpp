#ifndef TDLS_SOLVERS_TILED_LUPP_SOLVER_STATIC_HPP
#define TDLS_SOLVERS_TILED_LUPP_SOLVER_STATIC_HPP



/// \file
/// \brief TiledLUpp solver - LU with partial pivoting on tile grids, one
/// thread / work-item per system.
/// \author Tristan Chenaille
///
/// The matrix is split into a grid of TSxTS tiles; Gaussian elimination
/// runs on tiles instead of scalars. Only the tiles involved in the
/// current step live in registers; the full matrix stays in remote memory
/// (shared, global, or plain host memory). Pivoting is logical: piv[] maps
/// logical row -> physical row, rows are physically swapped only inside the
/// register-resident diagonal tile. When the best in-tile pivot falls
/// below TiledLUppSolverConfig::oot_threshold, the search extends below the tile
/// (out-of-tile pivoting) and candidate values are corrected on the fly
/// for the eliminations they have not received yet.
///
/// **Addressing convention (backend-agnostic, no accessor objects).**
/// The solver never sees a thread index: callers pre-offset every remote
/// pointer with the lane/system index and pass a runtime stride.
/// Element (r,c) of the matrix lives at `A[((r)*N+(c))*A_stride]`. The
/// pivot array and the right-hand sides follow the same convention. Each
/// of the three arrays has an *internal* residency mode (plain
/// caller-local array, stride ignored) selected by the `internal_rhs` /
/// `internal_piv` / `internal_matrix` template booleans - the
/// ternary on a constexpr bool costs nothing. The single form
/// `base[e*stride]` covers every batched layout without touching the
/// solver: AoS (stride 1), SoA (stride = batch size), AoSoA (stride = W).
///
/// **Tile grid.** F = N/TS full tiles per dimension, plus one trailing
/// tile of extent TAIL = N - F*TS when N is not a multiple of TS. Every
/// sweep is a runtime loop over the full tiles followed by an
/// `if constexpr (TAIL > 0)` epilogue instantiated with the trailing
/// extent; phantom slots of partial tiles are skipped at compile time by
/// the extent template parameters.
///
/// **Factored format.** The diagonal of the factored matrix holds the
/// RECIPROCALS of the U pivots (consumers multiply; each division is paid
/// once per pivot). This deliberately differs from the LAPACK `getrf`
/// convention - a factorization produced here must be consumed by the
/// substitution routines of this solver.
///
/// Originally developed and validated in the tfelGPU project.



#include <cmath>
#include <type_traits>

#include <tdls/solvers/tiled_lupp/config.hpp>
#include <tdls/solvers/tiled_lupp/tile_ops.hpp>



// gcc's flow analysis cannot prove that the phantom slots of trailing
// register tiles are never read (the extent-bounded loops skip them by
// construction, a property validated bitwise against reference solvers),
// and emits spurious -Wmaybe-uninitialized warnings on some
// internal-matrix instantiations at -O2. The suppression is scoped to
// this header and to that warning only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// clang reports a forced unrolling that the optimizer could not perform
// through -Wpass-failed. The unrolling requested by TDLS_UNROLL_FORCE is
// a performance hint: a failed hint does not affect correctness. The
// suppression is scoped to this header and to that warning only.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

namespace tdls {



/* Addressing macros. They expand inside member functions where the
   parameter names (A, A_stride, piv, piv_stride, x, b, y, rhs_stride,
   xcol_stride) and the residency template booleans are in scope.
   #undef'd at the end of this header. */

/// \def TDLS_LUPP_A
/// \brief Element (r, c) of the factor matrix: contiguous under internal
/// residency, strided otherwise.
#define TDLS_LUPP_A(r, c) A[internal_matrix ? ((r) * N + (c)) : ((r) * N + (c)) * A_stride]
/// \def TDLS_LUPP_PIV
/// \brief Pivot entry i: contiguous under internal residency, strided
/// otherwise.
#define TDLS_LUPP_PIV(i) piv[internal_piv ? (i) : (i) * piv_stride]
/// \def TDLS_LUPP_X
/// \brief Entry i of the solution vector: contiguous under internal
/// residency, strided otherwise.
#define TDLS_LUPP_X(i) x[internal_rhs ? (i) : (i) * rhs_stride]
/// \def TDLS_LUPP_B
/// \brief Entry i of the right-hand side: contiguous under internal
/// residency, strided otherwise.
#define TDLS_LUPP_B(i) b[internal_rhs ? (i) : (i) * rhs_stride]
/// \def TDLS_LUPP_XW
/// \brief Entry i of column w of a multi right-hand-side block: W
/// contiguous columns under internal residency, xcol_stride-strided
/// columns in remote memory. W = 1 collapses to TDLS_LUPP_X exactly.
#define TDLS_LUPP_XW(w, i) x[internal_rhs ? (w) * N + (i) : (i) * rhs_stride + (w) * xcol_stride]
/// \def TDLS_LUPP_Y
/// \brief Entry i of the fused right-hand side of solve_fused (y follows
/// the matrix rows through pivoting).
#define TDLS_LUPP_Y(i) y[internal_rhs ? (i) : (i) * rhs_stride]



/// \brief Tiled dense LU factorization with logical partial pivoting and
/// out-of-tile pivot recovery, solving one NxN system per call.
///
/// All entry points are static, host- and device-callable, and take raw
/// pointers pre-offset by the caller plus one runtime stride per array
/// (see the file documentation for the addressing convention). Entry
/// points:
///   - factorize:             A := P*L*U in place, pivots out (RL or LL)
///   - substitute:            x := U^-1 L^-1 P b (b and x distinct)
///   - substitute_canonical:  idem with b = e_col (tangent-operator columns)
///   - substitute_inplace:    idem with b == x (cycle-decomposition permute)
///   - solve:                 factorize + substitute
///   - solve_fused:           factorize with the forward pass folded in
///
/// factorize/substitute stay separate so a factorization can be reused
/// across several right-hand sides. Substitution is schedule-independent:
/// RL and LL produce the same L\\U layout.
///
/// The residency template booleans (internal_rhs / internal_piv /
/// internal_matrix) deliberately have no default: they describe what the
/// passed buffers ARE, so each call site must state them explicitly - a
/// silent default could mismatch the actual argument and corrupt results
/// without any diagnostic.
///
/// \tparam T                 scalar type (float or double)
/// \tparam N                 system dimension (N >= 2)
/// \tparam TiledLUppSolverConfig compile-time knobs - tile size, schedule,
///         pivoting thresholds, unroll policy; see TiledLUppDefaultConfig
///         and TiledLUppConfig
template<typename T, int N, typename TiledLUppSolverConfig = TiledLUppDefaultConfig<T>>
struct TiledLUppSolverStatic {

    static constexpr int TS = TiledLUppSolverConfig::tile_size; ///< tile size (int)
    static constexpr TiledLUppSchedule Schedule =
        TiledLUppSolverConfig::schedule; ///< elimination schedule (RightLooking or LeftLooking)

    static_assert(N >= 2, "TiledLUppSolverStatic: N must be >= 2");
    static_assert(TiledLUppSolverConfig::singular_eps <= TiledLUppSolverConfig::oot_threshold,
                  "TiledLUppSolverStatic: singular_eps must not exceed oot_threshold (the "
                  "floor applies to the out-of-tile recovery path)");
    static_assert(TS >= 2 && TS <= N, "TiledLUppSolverStatic: tile size must satisfy 2 <= TS <= N");

    static constexpr int F    = N / TS;     ///< full tiles per dimension
    static constexpr int TAIL = N - F * TS; ///< trailing tile extent (0 = divisible)

    /// \brief Tile micro-kernels instantiated for this configuration.
    using Ops = TiledLUppTileOps<T, TS, TiledLUppSolverConfig::unroll_inner>;

    /* =====================================================================
       Remote <-> register tile movement.
       `load/store_tile` go through a cached physical-row segment (the RL
       schedule keeps one per tile); `load/store_tile_piv` read the
       permutation inline, once per row (LL schedule and the substitution
       do this - it adds no integer registers).
       ===================================================================== */

    /// \brief Load an RxC tile through a cached physical-row segment.
    /// \tparam R tile row extent
    /// \tparam C tile column extent
    /// \tparam internal_matrix residency of A
    /// \param[in]  A        matrix (caller-pre-offset)
    /// \param[in]  A_stride element stride of A (external mode)
    /// \param[in]  prow     physical rows of the tile
    /// \param[in]  col0     first global column of the tile
    /// \param[out] t        destination register tile
    template<int R, int C, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT prow,
              const int col0, T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                TDLS_UNROLL_FORCE
                for (int j = 0; j < C; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(prow[i], col0 + j);
            }
        } else {
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(prow[i], col0 + j);
            }
        }
    }

    /// \brief Store an RxC tile through a cached physical-row segment.
    /// \tparam R tile row extent
    /// \tparam C tile column extent
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A (external mode)
    /// \param[in]     prow     physical rows of the tile
    /// \param[in]     col0     first global column of the tile
    /// \param[in]     t        source register tile
    template<int R, int C, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    store_tile(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT prow,
               const int col0, const T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                TDLS_UNROLL_FORCE
                for (int j = 0; j < C; ++j)
                    TDLS_LUPP_A(prow[i], col0 + j) = t[i * TS + j];
            }
        } else {
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j)
                    TDLS_LUPP_A(prow[i], col0 + j) = t[i * TS + j];
            }
        }
    }

    /// \brief Load an RxC tile, reading the permutation inline (one read
    /// per row).
    /// \tparam R tile row extent
    /// \tparam C tile column extent
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    template<int R, int C, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                  const int piv_stride, const int row0, const int col0, T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                TDLS_UNROLL_FORCE
                for (int j = 0; j < C; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        } else {
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                for (int j = 0; j < C; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        }
    }

    /// \brief Store an RxC tile, reading the permutation inline (one read
    /// per row).
    /// \tparam R tile row extent
    /// \tparam C tile column extent
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     row0       first global (logical) row of the tile
    /// \param[in]     col0       first global column of the tile
    /// \param[in]     t          source register tile
    template<int R, int C, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    store_tile_piv(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                   const int piv_stride, const int row0, const int col0, const T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                TDLS_UNROLL_FORCE
                for (int j = 0; j < C; ++j)
                    TDLS_LUPP_A(phys, col0 + j) = t[i * TS + j];
            }
        } else {
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                for (int j = 0; j < C; ++j)
                    TDLS_LUPP_A(phys, col0 + j) = t[i * TS + j];
            }
        }
    }

    /// \brief Triangular variant of the diagonal-tile load for the forward
    /// substitution: only the strict lower triangle of L\\U is read.
    ///
    /// The untouched slots of the register tile are never read -
    /// compile-time loop bounds, no branches, results bitwise identical.
    /// \tparam R tile extent
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    template<int R, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv_lower(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                        const int piv_stride, const int row0, const int col0, T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 1; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                TDLS_UNROLL_FORCE
                for (int j = 0; j < i; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        } else {
            for (int i = 1; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                for (int j = 0; j < i; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        }
    }

    /// \brief Triangular variant of the diagonal-tile load for the backward
    /// substitution: only the upper triangle including the diagonal is read.
    /// \tparam R tile extent
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    template<int R, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv_upper(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                        const int piv_stride, const int row0, const int col0, T* TDLS_RESTRICT t) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                TDLS_UNROLL_FORCE
                for (int j = i; j < R; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        } else {
            for (int i = 0; i < R; ++i) {
                const int phys = TDLS_LUPP_PIV(row0 + i);
                for (int j = i; j < R; ++j)
                    t[i * TS + j] = TDLS_LUPP_A(phys, col0 + j);
            }
        }
    }

    /* =====================================================================
       Diagonal-tile factorization with out-of-tile pivoting.
       Shared by both schedules; the only schedule-dependent part is the
       replay a candidate row needs before it can be compared:
         RL - the trailing matrix is already Schur-updated, so a candidate
              only misses the current tile's columns t < c.
         LL - rows below additionally miss every prior tile's L*U
              contribution (columns [0, k0)), replayed term by term.
       ===================================================================== */

    /// \brief One column step of the diagonal-tile factorization: pivot
    /// search (in-tile, then out-of-tile recovery), permutation update,
    /// row swap or cross-tile pull, column elimination.
    ///
    /// `tile` holds the (LL: prior-corrected) KExKE diagonal tile; physical
    /// row swaps happen in the register tile only.
    /// \tparam KE           extent of the diagonal tile
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \tparam fuse_rhs     apply the pivot swaps to the fused RHS y
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     k0         first global row/column of the tile
    /// \param[in,out] tile       register-resident diagonal tile
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in]     c          tile column to factor
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false when the matrix is singular at this column.
    template<int KE, bool internal_piv, bool internal_matrix, bool oot_diag, bool fuse_rhs,
             bool internal_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factor_diag_column(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                       const int piv_stride, const int k0, T* TDLS_RESTRICT tile, int& oot_count,
                       const int c, T* TDLS_RESTRICT y, const int rhs_stride) {

        const int gc = k0 + c; // global column

        // In-tile pivot search (rows c..KE of the register tile)
        int best_r = c;
        T best     = std::fabs(tile[c * TS + c]);
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int r = c + 1; r < KE; ++r) {
                const T v = std::fabs(tile[r * TS + c]);
                if (v > best) {
                    best   = v;
                    best_r = r;
                }
            }
        } else {
            for (int r = c + 1; r < KE; ++r) {
                const T v = std::fabs(tile[r * TS + c]);
                if (v > best) {
                    best   = v;
                    best_r = r;
                }
            }
        }

        int piv_row; // winning global (logical) row

        if (best >= TiledLUppSolverConfig::oot_threshold) {
            piv_row = k0 + best_r;
        } else if constexpr (KE < TS) {
            // Trailing tile: no rows below to recover from. Diagnostic
            // order: singularity verdict first, then count the weak pivot
            // (full tiles count before the verdict).
            if (best < TiledLUppSolverConfig::singular_eps) return false;
            if constexpr (oot_diag) ++oot_count;
            piv_row = k0 + best_r;
        } else {
            // Out-of-tile recovery: scan the rows below the tile and
            // evaluate each candidate as if it had received the
            // eliminations it is missing, keeping the best (or, with
            // TiledLUppSolverConfig::oot_first_acceptable, the first to reach the threshold).
            // Cold path - never unroll-annotated.
            if constexpr (oot_diag) ++oot_count;
            T gbest       = best;
            int gbest_row = k0 + best_r;

            for (int row = k0 + KE; row < N; ++row) {
                const int phys = TDLS_LUPP_PIV(row);

                T corrected = TDLS_LUPP_A(phys, gc);
                if constexpr (Schedule == TiledLUppSchedule::LeftLooking) {
                    for (int bj0 = 0; bj0 < k0; bj0 += TS)
                        for (int p = 0; p < TS; ++p)
                            corrected -= TDLS_LUPP_A(phys, bj0 + p) *
                                         TDLS_LUPP_A(TDLS_LUPP_PIV(bj0 + p), gc);
                }

                if (c > 0) {
                    T L_row[TS];
                    for (int t = 0; t < c; ++t) {
                        T a_t = TDLS_LUPP_A(phys, k0 + t);
                        if constexpr (Schedule == TiledLUppSchedule::LeftLooking) {
                            for (int bj0 = 0; bj0 < k0; bj0 += TS)
                                for (int p = 0; p < TS; ++p)
                                    a_t -= TDLS_LUPP_A(phys, bj0 + p) *
                                           TDLS_LUPP_A(TDLS_LUPP_PIV(bj0 + p), k0 + t);
                        }
                        for (int p = 0; p < t; ++p)
                            a_t -= L_row[p] * tile[p * TS + t];
                        L_row[t] = a_t * tile[t * TS + t]; // diag holds 1/pivot
                        corrected -= L_row[t] * tile[t * TS + c];
                    }
                }

                const T v = std::fabs(corrected);
                if (v > gbest) {
                    gbest     = v;
                    gbest_row = row;
                }

                // First-acceptable out-of-tile pivot: a candidate that
                // reaches the threshold already beats the sub-threshold
                // in-tile pivot, so stop scanning. The running max above is
                // kept as the fallback when no candidate is acceptable.
                if constexpr (TiledLUppSolverConfig::oot_first_acceptable)
                    if (v >= TiledLUppSolverConfig::oot_threshold) break;
            }

            if (gbest < TiledLUppSolverConfig::singular_eps) return false;
            piv_row = gbest_row;
        }

        {
            // Swap the permutation entries unconditionally: when
            // piv_row == gc the exchanges write back the same values
            // (bitwise no-ops), and removing the comparison removes a
            // divergent branch from the hot path (lanes of a warp pick
            // different pivots almost every column).
            const int tmp          = TDLS_LUPP_PIV(gc);
            TDLS_LUPP_PIV(gc)      = TDLS_LUPP_PIV(piv_row);
            TDLS_LUPP_PIV(piv_row) = tmp;

            if constexpr (fuse_rhs) {
                // The fused RHS follows the rows through pivoting. For an
                // internal y the swap is predicated over the unrolled index
                // range - y must never be dynamically indexed or it is
                // demoted to local memory.
                if constexpr (internal_rhs) {
                    if constexpr (TiledLUppSolverConfig::unroll_inner) {
                        TDLS_UNROLL_FORCE
                        for (int r = 0; r < N; ++r) {
                            if (r == piv_row) {
                                const T ty      = TDLS_LUPP_Y(gc);
                                TDLS_LUPP_Y(gc) = TDLS_LUPP_Y(r);
                                TDLS_LUPP_Y(r)  = ty;
                            }
                        }
                    } else {
                        for (int r = 0; r < N; ++r) {
                            if (r == piv_row) {
                                const T ty      = TDLS_LUPP_Y(gc);
                                TDLS_LUPP_Y(gc) = TDLS_LUPP_Y(r);
                                TDLS_LUPP_Y(r)  = ty;
                            }
                        }
                    }
                } else {
                    const T ty           = TDLS_LUPP_Y(gc);
                    TDLS_LUPP_Y(gc)      = TDLS_LUPP_Y(piv_row);
                    TDLS_LUPP_Y(piv_row) = ty;
                }
            }

            if (piv_row < k0 + KE) {
                Ops::swap_rows(tile, c, piv_row - k0);
            } else {
                // Cross-tile swap: pull the new row into the tile and
                // replay everything it missed - prior tiles (LL), then
                // the current tile's factored columns, on the FULL row.
                const int phys = TDLS_LUPP_PIV(gc);
                if constexpr (TiledLUppSolverConfig::unroll_inner) {
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < KE; ++j) {
                        tile[c * TS + j] = TDLS_LUPP_A(phys, k0 + j);
                        if constexpr (Schedule == TiledLUppSchedule::LeftLooking) {
                            for (int bj0 = 0; bj0 < k0; bj0 += TS)
                                for (int p = 0; p < TS; ++p)
                                    tile[c * TS + j] -= TDLS_LUPP_A(phys, bj0 + p) *
                                                        TDLS_LUPP_A(TDLS_LUPP_PIV(bj0 + p), k0 + j);
                        }
                    }
                } else {
                    for (int j = 0; j < KE; ++j) {
                        tile[c * TS + j] = TDLS_LUPP_A(phys, k0 + j);
                        if constexpr (Schedule == TiledLUppSchedule::LeftLooking) {
                            for (int bj0 = 0; bj0 < k0; bj0 += TS)
                                for (int p = 0; p < TS; ++p)
                                    tile[c * TS + j] -= TDLS_LUPP_A(phys, bj0 + p) *
                                                        TDLS_LUPP_A(TDLS_LUPP_PIV(bj0 + p), k0 + j);
                        }
                    }
                }

                if (c > 0) {
                    T L_row[TS];
                    for (int t = 0; t < c; ++t) {
                        T a_t = tile[c * TS + t];
                        for (int p = 0; p < t; ++p)
                            a_t -= L_row[p] * tile[p * TS + t];
                        L_row[t]         = a_t * tile[t * TS + t]; // diag holds 1/pivot
                        tile[c * TS + t] = L_row[t];
                    }
                    for (int j = c; j < KE; ++j) {
                        for (int t = 0; t < c; ++t)
                            tile[c * TS + j] -= L_row[t] * tile[t * TS + j];
                    }
                }
            }
        }

        Ops::template eliminate_column<KE, KE>(tile, c);
        return true;
    }

    /// \brief Factor the KExKE diagonal tile in registers, with
    /// out-of-tile pivot recovery (drives the per-column loop of
    /// factor_diag_column).
    /// \tparam KE           extent of the diagonal tile
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \tparam fuse_rhs     apply the pivot swaps to the fused RHS y
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     k0         first global row/column of the tile
    /// \param[in,out] tile       register-resident diagonal tile, loaded
    ///                (and, LL: prior-corrected) by the caller
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false on a singular matrix.
    template<int KE, bool internal_piv, bool internal_matrix, bool oot_diag, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factor_diag_tile(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                     const int piv_stride, const int k0, T* TDLS_RESTRICT tile, int& oot_count,
                     T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int c = 0; c < KE; ++c) {
                if (!factor_diag_column<KE, internal_piv, internal_matrix, oot_diag, fuse_rhs,
                                        internal_rhs>(A, A_stride, piv, piv_stride, k0, tile,
                                                      oot_count, c, y, rhs_stride))
                    return false;
            }
        } else {
            for (int c = 0; c < KE; ++c) {
                if (!factor_diag_column<KE, internal_piv, internal_matrix, oot_diag, fuse_rhs,
                                        internal_rhs>(A, A_stride, piv, piv_stride, k0, tile,
                                                      oot_count, c, y, rhs_stride))
                    return false;
            }
        }
        return true;
    }

    /* =====================================================================
       RIGHT-LOOKING schedule. Step k: factor the diagonal tile, then push
       its factors into the trailing matrix (TRSM right over the row panel,
       TRSM down + Schur complement over the rows below). The physical-row
       segments pk/pi are cached per tile (they fold to nothing when the
       permutation itself is internal).
       ===================================================================== */

    /// \brief RL: one row-panel tile update, Akj := L^-1 Akj.
    /// \tparam KE extent of the factored diagonal tile
    /// \tparam JE column extent of the updated tile
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A (external mode)
    /// \param[in]     pk       physical rows of the diagonal block row
    /// \param[in]     tile     factored diagonal tile
    /// \param[in]     j0       first global column of the updated tile
    template<int KE, int JE, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_trsm_right_one(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT pk,
                      const T* TDLS_RESTRICT tile, const int j0) {
        T Akj[TS * TS];
        load_tile<KE, JE, internal_matrix>(A, A_stride, pk, j0, Akj);
        Ops::template trsm_left_unit<KE, JE>(tile, Akj);
        store_tile<KE, JE, internal_matrix>(A, A_stride, pk, j0, Akj);
    }

    /// \brief RL: one Schur-complement tile update, Aij -= Aik * Akj,
    /// streaming the factored Akj row by row from remote memory (it is not
    /// worth a third register tile).
    /// \tparam KE inner extent (current step)
    /// \tparam IE row extent of the updated tile
    /// \tparam JE column extent of the updated tile
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A (external mode)
    /// \param[in]     pk       physical rows of the factored block row
    /// \param[in]     pi       physical rows of the updated block row
    /// \param[in]     Aik      register L panel of the updated block row
    /// \param[in]     j0       first global column of the updated tile
    template<int KE, int IE, int JE, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_schur_one(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT pk,
                 const int* TDLS_RESTRICT pi, const T* TDLS_RESTRICT Aik, const int j0) {
        T Aij[TS * TS];
        load_tile<IE, JE, internal_matrix>(A, A_stride, pi, j0, Aij);

        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int p = 0; p < KE; ++p) {
                T Akj_row[TS];
                TDLS_UNROLL_FORCE
                for (int j = 0; j < JE; ++j)
                    Akj_row[j] = TDLS_LUPP_A(pk[p], j0 + j);
                TDLS_UNROLL_FORCE
                for (int i = 0; i < IE; ++i) {
                    const T L_ip = Aik[i * TS + p];
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < JE; ++j)
                        Aij[i * TS + j] -= L_ip * Akj_row[j];
                }
            }
        } else {
            for (int p = 0; p < KE; ++p) {
                T Akj_row[TS];
                for (int j = 0; j < JE; ++j)
                    Akj_row[j] = TDLS_LUPP_A(pk[p], j0 + j);
                for (int i = 0; i < IE; ++i) {
                    const T L_ip = Aik[i * TS + p];
                    for (int j = 0; j < JE; ++j)
                        Aij[i * TS + j] -= L_ip * Akj_row[j];
                }
            }
        }

        store_tile<IE, JE, internal_matrix>(A, A_stride, pi, j0, Aij);
    }

    /// \brief RL: TRSM down + Schur sweep of one row block below the
    /// diagonal, with the optional fused forward-substitution push.
    /// \tparam KE extent of the diagonal tile
    /// \tparam IE row extent of the updated block row
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam fuse_rhs     push the fused RHS y along (solve_fused)
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     pk         physical rows of the diagonal tile
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in]     i0         first global row of the updated block row
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    template<int KE, int IE, bool internal_piv, bool internal_matrix, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_update_row_one(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                      const int piv_stride, const int* TDLS_RESTRICT pk,
                      const T* TDLS_RESTRICT tile, const int k, const int i0,
                      T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        const int k0 = k * TS;

        int pi[TS];
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < IE; ++i)
                pi[i] = TDLS_LUPP_PIV(i0 + i);
        } else {
            for (int i = 0; i < IE; ++i)
                pi[i] = TDLS_LUPP_PIV(i0 + i);
        }

        // TRSM down: Aik := Aik * U^-1
        T Aik[TS * TS];
        load_tile<IE, KE, internal_matrix>(A, A_stride, pi, k0, Aik);
        Ops::template trsm_right<KE, IE>(tile, Aik);
        store_tile<IE, KE, internal_matrix>(A, A_stride, pi, k0, Aik);

        // Fused forward substitution: push the solved y_k segment into this
        // row block while its L panel sits in registers (this is the whole
        // point of solve_fused - the separate forward pass reloads vanish).
        if constexpr (fuse_rhs) {
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int r = 0; r < IE; ++r) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < KE; ++j)
                        sum += Aik[r * TS + j] * TDLS_LUPP_Y(k0 + j);
                    TDLS_LUPP_Y(i0 + r) -= sum;
                }
            } else {
                for (int r = 0; r < IE; ++r) {
                    T sum = T(0);
                    for (int j = 0; j < KE; ++j)
                        sum += Aik[r * TS + j] * TDLS_LUPP_Y(k0 + j);
                    TDLS_LUPP_Y(i0 + r) -= sum;
                }
            }
        }

        // Schur sweep over the trailing columns
        for (int j = k + 1; j < F; ++j)
            rl_schur_one<KE, IE, TS, internal_matrix>(A, A_stride, pk, pi, Aik, j * TS);
        if constexpr (TAIL > 0 && KE == TS)
            rl_schur_one<KE, IE, TAIL, internal_matrix>(A, A_stride, pk, pi, Aik, F * TS);
    }

    /// \brief RL: one full factorization step (diagonal tile + trailing
    /// updates).
    /// \tparam KE           extent of the diagonal tile of this step
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \tparam fuse_rhs     apply the step to the fused RHS y (solve_fused)
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false on a singular matrix.
    template<int KE, bool internal_piv, bool internal_matrix, bool oot_diag, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    rl_step(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
            const int k, int& oot_count, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        const int k0 = k * TS;

        T tile[TS * TS];
        load_tile_piv<KE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                             tile);

        if (!factor_diag_tile<KE, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, k0, tile, oot_count, y, rhs_stride))
            return false;

        // Physical rows of the tile after the swaps of this step
        int pk[TS];
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < KE; ++i)
                pk[i] = TDLS_LUPP_PIV(k0 + i);
        } else {
            for (int i = 0; i < KE; ++i)
                pk[i] = TDLS_LUPP_PIV(k0 + i);
        }

        store_tile<KE, KE, internal_matrix>(A, A_stride, pk, k0, tile);

        // Fused forward substitution: this tile's y segment is final from
        // here on - unit-lower-solve it while the tile is in registers.
        if constexpr (fuse_rhs) {
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int kk = 0; kk < KE; ++kk) {
                    TDLS_UNROLL_FORCE
                    for (int i = kk + 1; i < KE; ++i)
                        TDLS_LUPP_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_Y(k0 + kk);
                }
            } else {
                for (int kk = 0; kk < KE; ++kk) {
                    for (int i = kk + 1; i < KE; ++i)
                        TDLS_LUPP_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_Y(k0 + kk);
                }
            }
        }

        // TRSM right over the row panel
        for (int j = k + 1; j < F; ++j)
            rl_trsm_right_one<KE, TS, internal_matrix>(A, A_stride, pk, tile, j * TS);
        if constexpr (TAIL > 0 && KE == TS)
            rl_trsm_right_one<KE, TAIL, internal_matrix>(A, A_stride, pk, tile, F * TS);

        // TRSM down + Schur over the rows below
        for (int i = k + 1; i < F; ++i)
            rl_update_row_one<KE, TS, internal_piv, internal_matrix, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, pk, tile, k, i * TS, y, rhs_stride);
        if constexpr (TAIL > 0 && KE == TS)
            rl_update_row_one<KE, TAIL, internal_piv, internal_matrix, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, pk, tile, k, F * TS, y, rhs_stride);
        return true;
    }

    /* =====================================================================
       LEFT-LOOKING schedule. Step k: pull every prior tile's L*U
       contribution into the current block column/row, then factor the
       diagonal tile and triangular-solve its panels. Each tile of the
       trailing matrix is written once per factorization (vs once per step
       for RL) at the cost of replaying prior tiles on each visit. The
       permutation is read inline (one read per row), never cached in
       segments.
       ===================================================================== */

    /// \brief LL: t (RExCE, rows row0.., cols col0..) -= sum over prior tiles
    /// bj < k0/TS of L(row0.., bj) * U(bj, col0..). All prior tiles are full.
    /// \tparam RE row extent of the corrected tile
    /// \tparam CE column extent of the corrected tile
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     row0       first global row of the corrected tile
    /// \param[in]     col0       first global column of the corrected tile
    /// \param[in]     k0         first row/column of the current step
    /// \param[in,out] t          register tile being corrected
    template<int RE, int CE, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_correct_tile(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                    const int piv_stride, const int row0, const int col0, const int k0,
                    T* TDLS_RESTRICT t) {
        for (int bj0 = 0; bj0 < k0; bj0 += TS) {
            T Lt[TS * TS];
            load_tile_piv<RE, TS, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, row0,
                                                                 bj0, Lt);
            T Ut[TS * TS];
            load_tile_piv<TS, CE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, bj0,
                                                                 col0, Ut);
            Ops::template gemm_sub<RE, CE, TS>(t, Lt, Ut);
        }
    }

    /// \brief LL: correct + TRSM one L-panel tile below the diagonal, with
    /// the optional fused forward-substitution push.
    /// \tparam KE extent of the diagonal tile
    /// \tparam IE row extent of the updated tile
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam fuse_rhs     push the fused RHS y along (solve_fused)
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k0         first global row/column of the step
    /// \param[in]     i0         first global row of the updated tile
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    template<int KE, int IE, bool internal_piv, bool internal_matrix, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_update_below_one(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                        const int piv_stride, const T* TDLS_RESTRICT tile, const int k0,
                        const int i0, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        T B[TS * TS];
        load_tile_piv<IE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, i0, k0,
                                                             B);
        ll_correct_tile<IE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, i0, k0,
                                                               k0, B);
        Ops::template trsm_right<KE, IE>(tile, B);
        store_tile_piv<IE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, i0, k0,
                                                              B);

        // Fused forward substitution: B is the final L(i,k) panel - push the
        // solved y_k segment into this row block while it is in registers.
        if constexpr (fuse_rhs) {
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int r = 0; r < IE; ++r) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < KE; ++j)
                        sum += B[r * TS + j] * TDLS_LUPP_Y(k0 + j);
                    TDLS_LUPP_Y(i0 + r) -= sum;
                }
            } else {
                for (int r = 0; r < IE; ++r) {
                    T sum = T(0);
                    for (int j = 0; j < KE; ++j)
                        sum += B[r * TS + j] * TDLS_LUPP_Y(k0 + j);
                    TDLS_LUPP_Y(i0 + r) -= sum;
                }
            }
        }
    }

    /// \brief LL: correct + TRSM one U-panel tile right of the diagonal.
    /// \tparam KE extent of the diagonal tile
    /// \tparam JE column extent of the updated tile
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k0         first global row/column of the step
    /// \param[in]     j0         first global column of the updated tile
    template<int KE, int JE, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_update_right_one(T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                        const int piv_stride, const T* TDLS_RESTRICT tile, const int k0,
                        const int j0) {
        T B[TS * TS];
        load_tile_piv<KE, JE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, j0,
                                                             B);
        ll_correct_tile<KE, JE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, j0,
                                                               k0, B);
        Ops::template trsm_left_unit<KE, JE>(tile, B);
        store_tile_piv<KE, JE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, j0,
                                                              B);
    }

    /// \brief LL: one full factorization step (correct + factor the
    /// diagonal tile, then its L and U panels).
    /// \tparam KE           extent of the diagonal tile of this step
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \tparam fuse_rhs     apply the step to the fused RHS y (solve_fused)
    /// \tparam internal_rhs residency of y
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false on a singular matrix.
    template<int KE, bool internal_piv, bool internal_matrix, bool oot_diag, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    ll_step(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
            const int k, int& oot_count, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        const int k0 = k * TS;

        T tile[TS * TS];
        load_tile_piv<KE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                             tile);
        ll_correct_tile<KE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                               k0, tile);

        if (!factor_diag_tile<KE, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, k0, tile, oot_count, y, rhs_stride))
            return false;

        store_tile_piv<KE, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                              tile);

        // Fused forward substitution: this tile's y segment is final from
        // here on - unit-lower-solve it while the tile is in registers.
        if constexpr (fuse_rhs) {
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int kk = 0; kk < KE; ++kk) {
                    TDLS_UNROLL_FORCE
                    for (int i = kk + 1; i < KE; ++i)
                        TDLS_LUPP_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_Y(k0 + kk);
                }
            } else {
                for (int kk = 0; kk < KE; ++kk) {
                    for (int i = kk + 1; i < KE; ++i)
                        TDLS_LUPP_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_Y(k0 + kk);
                }
            }
        }

        // L panel below the diagonal
        for (int i = k + 1; i < F; ++i)
            ll_update_below_one<KE, TS, internal_piv, internal_matrix, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, tile, k0, i * TS, y, rhs_stride);
        if constexpr (TAIL > 0 && KE == TS)
            ll_update_below_one<KE, TAIL, internal_piv, internal_matrix, fuse_rhs, internal_rhs>(
                A, A_stride, piv, piv_stride, tile, k0, F * TS, y, rhs_stride);

        // U row panel right of the diagonal
        for (int j = k + 1; j < F; ++j)
            ll_update_right_one<KE, TS, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride,
                                                                       tile, k0, j * TS);
        if constexpr (TAIL > 0 && KE == TS)
            ll_update_right_one<KE, TAIL, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, tile, k0, F * TS);
        return true;
    }

    /* =====================================================================
       FACTORIZE - public entry point.
       ===================================================================== */

    /// \brief Factor A := P*L*U in place.
    ///
    /// A becomes L\\U (unit lower L below the diagonal, U above including
    /// it, the diagonal holding the pivot RECIPROCALS), under logical row
    /// permutation piv (piv[i] = physical row holding logical row i).
    ///
    /// \tparam internal_piv true: piv is a caller-local int[N], stride
    ///         ignored; false: piv is remote, indexed with piv_stride
    /// \tparam internal_matrix true: A is a caller-local T[N*N], stride
    ///         ignored; false: A is remote, indexed with A_stride
    /// \tparam oot_diag when false, the out-of-tile diagnostics are
    ///         compiled out entirely (no counter register, no increments,
    ///         oot_count untouched) - use the overload without the
    ///         out-parameter
    /// \param[in,out] A          matrix, pre-offset by the caller
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \tparam fuse_rhs internal hook of solve_fused: folds the forward
    ///         substitution of y into the factorization
    /// \tparam internal_rhs residency of the fused y (meaningful with
    ///         fuse_rhs only)
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false on a singular matrix.
    template<bool internal_piv, bool internal_matrix, bool oot_diag = true, bool fuse_rhs = false,
             bool internal_rhs = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factorize(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
              int& oot_count, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {

        if constexpr (oot_diag) oot_count = 0;

        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < N; ++i)
                TDLS_LUPP_PIV(i) = i;
        } else {
            for (int i = 0; i < N; ++i)
                TDLS_LUPP_PIV(i) = i;
        }

        if constexpr (Schedule == TiledLUppSchedule::RightLooking) {
            for (int k = 0; k < F; ++k)
                if (!rl_step<TS, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                        A, A_stride, piv, piv_stride, k, oot_count, y, rhs_stride))
                    return false;
            if constexpr (TAIL > 0)
                if (!rl_step<TAIL, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                        A, A_stride, piv, piv_stride, F, oot_count, y, rhs_stride))
                    return false;
        } else {
            for (int k = 0; k < F; ++k)
                if (!ll_step<TS, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                        A, A_stride, piv, piv_stride, k, oot_count, y, rhs_stride))
                    return false;
            if constexpr (TAIL > 0)
                if (!ll_step<TAIL, internal_piv, internal_matrix, oot_diag, fuse_rhs, internal_rhs>(
                        A, A_stride, piv, piv_stride, F, oot_count, y, rhs_stride))
                    return false;
        }

        return true;
    }

    /// \brief Diagnostics-free factorize overload: no out-of-tile
    /// out-parameter at all.
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A          matrix, pre-offset by the caller
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \return false on a singular matrix.
    template<bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool factorize(T* TDLS_RESTRICT A, const int A_stride,
                                                            int* TDLS_RESTRICT piv,
                                                            const int piv_stride) {
        int unused = 0;
        return factorize<internal_piv, internal_matrix, false>(A, A_stride, piv, piv_stride,
                                                               unused);
    }

    /* =====================================================================
       SUBSTITUTION - schedule-independent.
       Forward: unit-lower solve tile by tile, each solved segment pushed
       into the tiles below it. Backward: each segment first pulls the
       trailing contributions, then upper-solves its diagonal tile.
       ===================================================================== */

    /// \brief Forward push: subtract L(m,k) * x_k from the x_m segment,
    /// for W columns at once.
    /// \tparam KE extent of the solved segment's tile
    /// \tparam ME extent of the target segment's tile
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k0          first global row of the solved segment
    /// \param[in]     m0          first global row of the target segment
    template<int KE, int ME, int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_push_one(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                 const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride,
                 const int xcol_stride, const int k0, const int m0) {
        T Lmk[TS * TS];
        load_tile_piv<ME, KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, m0, k0,
                                                             Lmk);
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < ME; ++i) {
                TDLS_UNROLL_FORCE
                for (int w = 0; w < W; ++w) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < KE; ++j)
                        sum += Lmk[i * TS + j] * TDLS_LUPP_XW(w, k0 + j);
                    TDLS_LUPP_XW(w, m0 + i) -= sum;
                }
            }
        } else {
            for (int i = 0; i < ME; ++i) {
                for (int w = 0; w < W; ++w) {
                    T sum = T(0);
                    for (int j = 0; j < KE; ++j)
                        sum += Lmk[i * TS + j] * TDLS_LUPP_XW(w, k0 + j);
                    TDLS_LUPP_XW(w, m0 + i) -= sum;
                }
            }
        }
    }

    /// \brief Forward step: unit-lower solve the diagonal tile's segment,
    /// then push it into the tiles below.
    /// \tparam KE extent of the diagonal tile of the step
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k           step index (k0 = k*TS)
    template<int KE, int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_step(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
             const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride,
             const int k) {
        const int k0 = k * TS;

        T Lkk[TS * TS];
        load_tile_piv_lower<KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                               Lkk);

        // In-tile unit-lower solve
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int kk = 0; kk < KE; ++kk) {
                TDLS_UNROLL_FORCE
                for (int i = kk + 1; i < KE; ++i) {
                    TDLS_UNROLL_FORCE
                    for (int w = 0; w < W; ++w)
                        TDLS_LUPP_XW(w, k0 + i) -= Lkk[i * TS + kk] * TDLS_LUPP_XW(w, k0 + kk);
                }
            }
        } else {
            for (int kk = 0; kk < KE; ++kk) {
                for (int i = kk + 1; i < KE; ++i) {
                    for (int w = 0; w < W; ++w)
                        TDLS_LUPP_XW(w, k0 + i) -= Lkk[i * TS + kk] * TDLS_LUPP_XW(w, k0 + kk);
                }
            }
        }

        // Push into the tiles below
        for (int m = k + 1; m < F; ++m)
            fwd_push_one<KE, TS, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, m * TS);
        if constexpr (TAIL > 0 && KE == TS)
            fwd_push_one<KE, TAIL, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, F * TS);
    }

    /// \brief Backward pull: subtract U(k,m) * x_m from the x_k segment,
    /// for W columns at once.
    /// \tparam KE extent of the updated segment's tile
    /// \tparam ME extent of the trailing segment's tile
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k0          first global row of the updated segment
    /// \param[in]     m0          first global row of the trailing segment
    template<int KE, int ME, int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_pull_one(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                 const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride,
                 const int xcol_stride, const int k0, const int m0) {
        T Ukm[TS * TS];
        load_tile_piv<KE, ME, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, m0,
                                                             Ukm);
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < KE; ++i) {
                TDLS_UNROLL_FORCE
                for (int w = 0; w < W; ++w) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < ME; ++j)
                        sum += Ukm[i * TS + j] * TDLS_LUPP_XW(w, m0 + j);
                    TDLS_LUPP_XW(w, k0 + i) -= sum;
                }
            }
        } else {
            for (int i = 0; i < KE; ++i) {
                for (int w = 0; w < W; ++w) {
                    T sum = T(0);
                    for (int j = 0; j < ME; ++j)
                        sum += Ukm[i * TS + j] * TDLS_LUPP_XW(w, m0 + j);
                    TDLS_LUPP_XW(w, k0 + i) -= sum;
                }
            }
        }
    }

    /// \brief Backward step: pull the trailing contributions, then
    /// upper-solve the diagonal tile's segment.
    /// \tparam KE extent of the diagonal tile of the step
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k           step index (k0 = k*TS)
    template<int KE, int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_step(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
             const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride,
             const int k) {
        const int k0 = k * TS;

        // Pull the trailing contributions
        for (int m = k + 1; m < F; ++m)
            bwd_pull_one<KE, TS, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, m * TS);
        if constexpr (TAIL > 0 && KE == TS)
            bwd_pull_one<KE, TAIL, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, F * TS);

        // In-tile upper solve
        T Ukk[TS * TS];
        load_tile_piv_upper<KE, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, k0, k0,
                                                               Ukk);

        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int kk = KE - 1; kk >= 0; --kk) {
                TDLS_UNROLL_FORCE
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_XW(w, k0 + kk) *= Ukk[kk * TS + kk]; // diag holds 1/pivot
                TDLS_UNROLL_FORCE
                for (int i = 0; i < kk; ++i) {
                    TDLS_UNROLL_FORCE
                    for (int w = 0; w < W; ++w)
                        TDLS_LUPP_XW(w, k0 + i) -= Ukk[i * TS + kk] * TDLS_LUPP_XW(w, k0 + kk);
                }
            }
        } else {
            for (int kk = KE - 1; kk >= 0; --kk) {
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_XW(w, k0 + kk) *= Ukk[kk * TS + kk]; // diag holds 1/pivot
                for (int i = 0; i < kk; ++i) {
                    for (int w = 0; w < W; ++w)
                        TDLS_LUPP_XW(w, k0 + i) -= Ukk[i * TS + kk] * TDLS_LUPP_XW(w, k0 + kk);
                }
            }
        }
    }

    /// \brief Backward pass alone - used by fwd_bwd and by solve_fused
    /// (whose forward pass happens inside the factorization).
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    template<int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_only(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
             const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride,
             const int xcol_stride) {
        if constexpr (TAIL > 0)
            bwd_step<TAIL, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, F);
        for (int k = F - 1; k >= 0; --k)
            bwd_step<TS, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k);
    }

    /// \brief Triangular solves on already-permuted column(s) x. W columns
    /// are processed per tile visit, so every L/U tile is loaded once for
    /// the whole block instead of once per column (W=1 = the single-RHS
    /// case).
    /// \tparam W  number of columns processed together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A (external mode)
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv (external mode)
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x (external mode)
    /// \param[in]     xcol_stride element stride between columns of x
    template<int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_bwd(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
            const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride) {
        for (int k = 0; k < F; ++k)
            fwd_step<TS, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k);
        if constexpr (TAIL > 0)
            fwd_step<TAIL, W, internal_rhs, internal_piv, internal_matrix>(
                A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, F);

        bwd_only<W, internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, x,
                                                                 rhs_stride, xcol_stride);
    }

    /* =====================================================================
       SUBSTITUTION - public entry points.
       ===================================================================== */

    /// \brief Solve x := U^-1 L^-1 P b from a prior factorize. b and x must
    /// not alias (use substitute_inplace for the aliased single-buffer
    /// pattern).
    ///
    /// \tparam internal_rhs true: b and x are caller-local T[N], strides
    ///         ignored; false: remote, indexed with rhs_stride
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A          factored matrix produced by factorize
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        permutation produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  b          right-hand side, in original (unpermuted) order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x (external mode)
    template<bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
               const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x,
               const int rhs_stride) {
        if constexpr (internal_rhs) {
            // Predicated gather: b[piv[i]] would index the internal b with
            // a runtime value and demote it to local memory. The equality
            // sweep keeps every index compile-time (measured: local memory
            // eliminated, identical values).
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int i = 0; i < N; ++i) {
                    const int p = TDLS_LUPP_PIV(i);
                    T v         = T(0);
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < N; ++j)
                        if (p == j) v = b[j];
                    TDLS_LUPP_X(i) = v;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    const int p = TDLS_LUPP_PIV(i);
                    T v         = T(0);
                    for (int j = 0; j < N; ++j)
                        if (p == j) v = b[j];
                    TDLS_LUPP_X(i) = v;
                }
            }
        } else {
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int i = 0; i < N; ++i)
                    TDLS_LUPP_X(i) = TDLS_LUPP_B(TDLS_LUPP_PIV(i));
            } else {
                for (int i = 0; i < N; ++i)
                    TDLS_LUPP_X(i) = TDLS_LUPP_B(TDLS_LUPP_PIV(i));
            }
        }
        fwd_bwd<1, internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, x,
                                                                rhs_stride, 0);
    }

    /// \brief Solve with b = e_col generated on the fly - the
    /// consistent-tangent-operator path.
    ///
    /// Thin alias of the W-column variant below (W=1 collapses to exactly
    /// the single-column code, so there is no separate implementation to
    /// maintain).
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A          factored matrix produced by factorize
    /// \param[in]  A_stride   element stride of A (external mode)
    /// \param[in]  piv        permutation produced by factorize
    /// \param[in]  piv_stride element stride of piv (external mode)
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x (external mode)
    template<bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_canonical(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                         const int piv_stride, const int col, T* TDLS_RESTRICT x,
                         const int rhs_stride) {
        substitute_canonical_block<1, internal_rhs, internal_piv, internal_matrix>(
            A, A_stride, piv, piv_stride, col, x, rhs_stride, 0);
    }

    /// \brief W canonical columns e_col0..e_{col0+W-1} solved per tile
    /// visit: every L/U tile is loaded once for the block instead of once
    /// per column.
    ///
    /// Column w lives at x + w*N (internal) or x + w*xcol_stride (remote);
    /// per-column arithmetic is identical to W separate
    /// substitute_canonical calls, so results match them bitwise.
    /// \tparam W number of consecutive canonical columns solved together
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]  A           factored matrix produced by factorize
    /// \param[in]  A_stride    element stride of A (external mode)
    /// \param[in]  piv         permutation produced by factorize
    /// \param[in]  piv_stride  element stride of piv (external mode)
    /// \param[in]  col0        index of the first canonical column
    /// \param[out] x           W solution columns
    /// \param[in]  rhs_stride  element stride of x (external mode)
    /// \param[in]  xcol_stride element stride between columns of x
    ///             (external mode)
    template<int W, bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_canonical_block(const T* TDLS_RESTRICT A, const int A_stride,
                               const int* TDLS_RESTRICT piv, const int piv_stride, const int col0,
                               T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride) {
        if constexpr (TiledLUppSolverConfig::unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < N; ++i) {
                const int p = TDLS_LUPP_PIV(i);
                TDLS_UNROLL_FORCE
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_XW(w, i) = (p == col0 + w) ? T(1) : T(0);
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const int p = TDLS_LUPP_PIV(i);
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_XW(w, i) = (p == col0 + w) ? T(1) : T(0);
            }
        }
        fwd_bwd<W, internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, x,
                                                                rhs_stride, xcol_stride);
    }

    /// \brief Solve in place: x already holds the unpermuted RHS on entry
    /// and the solution on exit.
    ///
    /// The permutation is applied in place by cycle decomposition, each
    /// cycle rotated once from its smallest index. Up to N = 64 a bitmask
    /// tracks the visited rows (a single 32- or 64-bit register); beyond,
    /// a cycle-leader scan is used instead (zero storage, an extra integer
    /// walk of each orbit). Both paths move the same values in the same
    /// order.
    /// \tparam internal_rhs residency of x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in]     A          factored matrix produced by factorize
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[in]     piv        permutation produced by factorize
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] x          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of x (external mode)
    template<bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_inplace(const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
                       const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride) {
        if constexpr (N <= 64) {
            // Bitmask cycle decomposition: one visited bit per row, the
            // whole state in a single 32- or 64-bit register.
            using mask_t   = std::conditional_t<(N <= 32), unsigned, unsigned long long>;
            mask_t visited = mask_t(0);
            if constexpr (TiledLUppSolverConfig::unroll_inner) {
                TDLS_UNROLL_FORCE
                for (int s = 0; s < N; ++s) {
                    if ((visited >> s) & mask_t(1)) continue;
                    const T tmp = TDLS_LUPP_X(s);
                    int cur     = s;
                    int nxt     = TDLS_LUPP_PIV(cur);
                    while (nxt != s) {
                        TDLS_LUPP_X(cur) = TDLS_LUPP_X(nxt);
                        visited |= mask_t(1) << cur;
                        cur = nxt;
                        nxt = TDLS_LUPP_PIV(cur);
                    }
                    TDLS_LUPP_X(cur) = tmp;
                    visited |= mask_t(1) << cur;
                }
            } else {
                for (int s = 0; s < N; ++s) {
                    if ((visited >> s) & mask_t(1)) continue;
                    const T tmp = TDLS_LUPP_X(s);
                    int cur     = s;
                    int nxt     = TDLS_LUPP_PIV(cur);
                    while (nxt != s) {
                        TDLS_LUPP_X(cur) = TDLS_LUPP_X(nxt);
                        visited |= mask_t(1) << cur;
                        cur = nxt;
                        nxt = TDLS_LUPP_PIV(cur);
                    }
                    TDLS_LUPP_X(cur) = tmp;
                    visited |= mask_t(1) << cur;
                }
            }
        } else {
            // Cycle-leader scan (no visited storage): a cycle is rotated
            // only when reached from its smallest index, detected by
            // walking the orbit. Values and order match the mask path.
            for (int s = 0; s < N; ++s) {
                int probe = TDLS_LUPP_PIV(s);
                while (probe > s)
                    probe = TDLS_LUPP_PIV(probe);
                if (probe != s) continue;

                const T tmp = TDLS_LUPP_X(s);
                int cur     = s;
                int nxt     = TDLS_LUPP_PIV(cur);
                while (nxt != s) {
                    TDLS_LUPP_X(cur) = TDLS_LUPP_X(nxt);
                    cur              = nxt;
                    nxt              = TDLS_LUPP_PIV(cur);
                }
                TDLS_LUPP_X(cur) = tmp;
            }
        }
        fwd_bwd<1, internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, x,
                                                                rhs_stride, 0);
    }

    /* =====================================================================
       SOLVE - convenience wrappers.
       ===================================================================== */

    /// \brief factorize + substitute in one call.
    /// \tparam internal_rhs residency of b and x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization (usable for
    ///                further substitute* calls)
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x (external mode)
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \return false on a singular matrix.
    template<bool internal_rhs, bool internal_piv, bool internal_matrix, bool oot_diag = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
          const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride, int& oot_count) {
        if (!factorize<internal_piv, internal_matrix, oot_diag>(A, A_stride, piv, piv_stride,
                                                                oot_count))
            return false;
        substitute<internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, b, x,
                                                                rhs_stride);
        return true;
    }

    /// \brief Diagnostics-free solve overload: no out-of-tile
    /// out-parameter at all.
    /// \tparam internal_rhs residency of b and x
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x (external mode)
    /// \return false on a singular matrix.
    template<bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv, const int piv_stride,
          const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride) {
        int unused = 0;
        return solve<internal_rhs, internal_piv, internal_matrix, false>(
            A, A_stride, piv, piv_stride, b, x, rhs_stride, unused);
    }

    /// \brief Factorization with the forward substitution folded in.
    ///
    /// y holds the unpermuted RHS on entry and the solution on exit. Each
    /// L panel is applied to y while it sits in registers during the
    /// factorization, so the separate forward pass - and all its tile
    /// reloads - disappears; only the backward pass remains. y follows the
    /// rows through pivoting (swap hook in factor_diag_column), which
    /// makes the result bitwise-identical to factorize +
    /// substitute_inplace.
    ///
    /// \tparam internal_rhs residency of y
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \tparam oot_diag     compile the out-of-tile counter in or out
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \return false on a singular matrix (y left partially updated).
    template<bool internal_rhs, bool internal_piv, bool internal_matrix, bool oot_diag = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve_fused(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride, int& oot_count) {
        if (!factorize<internal_piv, internal_matrix, oot_diag, true, internal_rhs>(
                A, A_stride, piv, piv_stride, oot_count, y, rhs_stride))
            return false;

        // Backward pass only - the forward one happened inside factorize.
        bwd_only<1, internal_rhs, internal_piv, internal_matrix>(A, A_stride, piv, piv_stride, y,
                                                                 rhs_stride, 0);
        return true;
    }

    /// \brief Diagnostics-free solve_fused overload: no out-of-tile
    /// out-parameter at all.
    /// \tparam internal_rhs residency of y
    /// \tparam internal_piv residency of piv
    /// \tparam internal_matrix residency of A
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A (external mode)
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv (external mode)
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y (external mode)
    /// \return false on a singular matrix (y left partially updated).
    template<bool internal_rhs, bool internal_piv, bool internal_matrix>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve_fused(T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride) {
        int unused = 0;
        return solve_fused<internal_rhs, internal_piv, internal_matrix, false>(
            A, A_stride, piv, piv_stride, y, rhs_stride, unused);
    }

    /* =====================================================================
       Note on pivot ownership: the pivot storage is ALWAYS caller-provided
       (internal array or remote scratch), for every entry point - one
       uniform calling convention. A solver-internal pivot would only be
       expressible for the single combination {solve/solve_fused x internal
       pivot}; externalizing it there costs nothing (the caller's int
       piv[N] inlines to the exact same codegen), so no special case is
       kept. For one-shot solves the contents are simply treated as
       scratch; for factorize + substitute* (consistent tangent), the same
       storage carries the permutation across the calls.
       ===================================================================== */
};



#undef TDLS_LUPP_A
#undef TDLS_LUPP_PIV
#undef TDLS_LUPP_X
#undef TDLS_LUPP_B
#undef TDLS_LUPP_XW
#undef TDLS_LUPP_Y



} // namespace tdls

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILED_LUPP_SOLVER_STATIC_HPP
