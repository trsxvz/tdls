#ifndef TDLS_SOLVERS_TILED_LUPP_SOLVER_DYNAMIC_HPP
#define TDLS_SOLVERS_TILED_LUPP_SOLVER_DYNAMIC_HPP



/// \file
/// \brief Runtime-size variant of the TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Same algorithm as TiledLUppSolverStatic (solver_static.hpp) - tiled LU with logical
/// partial pivoting, out-of-tile recovery, reciprocal-diagonal factored
/// format, right- and left-looking schedules - but the system dimension n
/// is a runtime function parameter instead of a template parameter. Only
/// the tile size (TiledLUppSolverConfig::tile_size) stays compile-time:
/// register tiles keep a fixed TSxTS footprint, while all loop bounds
/// over tiles and inside partial tiles are runtime values.
///
/// Deliberate differences with the compile-time solver:
///   - No unroll pragma anywhere: with runtime bounds nothing can be
///     register-resident by full unrolling, so the unroll_inner knob of
///     TiledLUppSolverConfig is ignored.
///   - No internal_rhs / internal_piv / internal_matrix booleans: without
///     unrolling, the internal residency mode degenerates into "external
///     with stride 1", so every array is plain pointer + stride. The
///     predicated register gathers/swaps of the static TiledLUpp solver become
///     direct indexed accesses (same values, simpler code).
///   - substitute_inplace dispatches at runtime (warp-uniform: n is the
///     same for every lane): a 64-bit visited bitmask up to n = 64, a
///     cycle-leader scan beyond - zero extra storage and no ceiling on n.
///
/// For equal shapes (same n, TS, schedule, TiledLUppSolverConfig thresholds), results are
/// bitwise identical to the compile-time solver: the arithmetic sequence
/// is the same, only addressing and loop mechanics differ.
///
/// The compile-time solver is the performance path; this variant is the
/// flexibility path (dimensions unknown at compile time, fast builds).
///
/// Preconditions: n >= 2. TS may exceed n (the grid is then a single
/// partial tile).



#include <cmath>

#include <tdls/solvers/tiled_lupp/config.hpp>



// gcc's flow analysis cannot prove that the replay buffer of the
// out-of-tile search is written before being read (the loop structure
// guarantees it, a property validated bitwise against the static
// solver), and emits spurious -Wmaybe-uninitialized warnings at -O2.
// The suppression is scoped to this header and to that warning only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

namespace tdls {



/* Addressing macros. They expand inside member functions where n and the
   parameter names (A, A_stride, piv, piv_stride, x, b, y, rhs_stride,
   xcol_stride) are in scope. #undef'd at the end of this header. */

#define TDLS_LUPP_DYN_A(r, c)  A[((r) * n + (c)) * A_stride]
#define TDLS_LUPP_DYN_PIV(i)   piv[(i) * piv_stride]
#define TDLS_LUPP_DYN_X(i)     x[(i) * rhs_stride]
#define TDLS_LUPP_DYN_B(i)     b[(i) * rhs_stride]
#define TDLS_LUPP_DYN_XW(w, i) x[(i) * rhs_stride + (w) * xcol_stride]
#define TDLS_LUPP_DYN_Y(i)     y[(i) * rhs_stride]



/// \brief Runtime-size tiled dense LU factorization with logical partial
/// pivoting and out-of-tile pivot recovery, solving one n x n system per
/// call.
///
/// All entry points are static, host- and device-callable, and take the
/// dimension n plus raw pointers pre-offset by the caller with one runtime
/// stride per array. Entry points:
///   - factorize:             A := P*L*U in place, pivots out (RL or LL)
///   - substitute:            x := U^-1 L^-1 P b (b and x distinct)
///   - substitute_canonical:  idem with b = e_col (tangent-operator columns)
///   - substitute_inplace:    idem with b == x (cycle-leader permute)
///   - solve:                 factorize + substitute
///   - solve_fused:           factorize with the forward pass folded in
///
/// The factored diagonal holds the RECIPROCALS of the U pivots, exactly as
/// in TiledLUppSolverStatic: a factorization produced here must be consumed by the
/// substitution routines of this family.
///
/// \tparam T                 scalar type (float or double)
/// \tparam TiledLUppSolverConfig compile-time knobs - tile size (may exceed
///         n), schedule, pivoting thresholds; see TiledLUppDefaultConfig and
///         TiledLUppConfig (unroll_inner is ignored by this variant)
template<typename T, typename TiledLUppSolverConfig = TiledLUppDefaultConfig<T>>
struct TiledLUppSolverDynamic {

    static constexpr int TS                  = TiledLUppSolverConfig::tile_size;
    static constexpr TiledLUppSchedule Sched = TiledLUppSolverConfig::schedule;

    static_assert(TiledLUppSolverConfig::singular_eps <= TiledLUppSolverConfig::oot_threshold,
                  "TiledLUppSolverDynamic: singular_eps must not exceed oot_threshold (the "
                  "floor applies to the out-of-tile recovery path)");
    static_assert(TS >= 2, "TiledLUppSolverDynamic: tile size must be >= 2");

    /// \brief Number of tiles per dimension (last one possibly partial).
    /// \param[in] n system dimension
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static int num_tiles(const int n) {
        return (n + TS - 1) / TS;
    }

    /// \brief Extent of the tile starting at row/column t0 (TS, or less
    /// for the trailing tile).
    /// \param[in] t0 first global row/column of the tile
    /// \param[in] n  system dimension
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static int tile_extent(const int t0, const int n) {
        return (n - t0 < TS) ? (n - t0) : TS;
    }

    /* =====================================================================
       Register-tile micro-kernels, runtime extents. A tile is a TSxTS
       array with row stride TS; the runtime extents bound the active part.
       ===================================================================== */

    /// \brief Row swap k <-> r inside the tile (direct, first ke columns).
    /// \param[in,out] t  register tile
    /// \param[in]     k  destination row (elimination column)
    /// \param[in]     r  source row to swap in
    /// \param[in]     ke active extent of the tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void ops_swap_rows(T* TDLS_RESTRICT t, const int k,
                                                                const int r, const int ke) {
        for (int j = 0; j < ke; ++j) {
            const T tmp   = t[k * TS + j];
            t[k * TS + j] = t[r * TS + j];
            t[r * TS + j] = tmp;
        }
    }

    /// \brief Gaussian elimination of column k inside the diagonal tile.
    ///
    /// Scales the sub-column by 1/pivot and updates the trailing block.
    /// The RECIPROCAL of the pivot is stored at the diagonal slot: every
    /// downstream consumer multiplies instead of dividing.
    /// \param[in,out] t  register tile
    /// \param[in]     k  column to eliminate
    /// \param[in]     re active row extent of the tile
    /// \param[in]     ce active column extent of the tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ops_eliminate_column(T* TDLS_RESTRICT t, const int k, const int re, const int ce) {
        const T inv_pivot = T(1) / t[k * TS + k];
        t[k * TS + k]     = inv_pivot;
        for (int i = k + 1; i < re; ++i) {
            t[i * TS + k] *= inv_pivot;
            for (int j = k + 1; j < ce; ++j)
                t[i * TS + j] -= t[i * TS + k] * t[k * TS + j];
        }
    }

    /// \brief B := L^-1 B, with L the unit lower part of the factored
    /// diagonal tile. L is kd x kd, B is kd x ce.
    /// \param[in]     lu factored diagonal tile (L\\U)
    /// \param[in,out] B  updated register tile
    /// \param[in]     kd extent of the factored diagonal tile
    /// \param[in]     ce column extent of B
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ops_trsm_left_unit(const T* TDLS_RESTRICT lu, T* TDLS_RESTRICT B, const int kd, const int ce) {
        for (int k = 0; k < kd; ++k) {
            for (int i = k + 1; i < kd; ++i) {
                const T L_ik = lu[i * TS + k];
                for (int j = 0; j < ce; ++j)
                    B[i * TS + j] -= L_ik * B[k * TS + j];
            }
        }
    }

    /// \brief B := B U^-1, with U the upper part of the factored diagonal
    /// tile. U is kd x kd, B is re x kd.
    ///
    /// The diagonal of `lu` already holds the pivot reciprocals - no
    /// divisions here.
    /// \param[in]     lu factored diagonal tile (L\\U, reciprocal diagonal)
    /// \param[in,out] B  updated register tile
    /// \param[in]     kd extent of the factored diagonal tile
    /// \param[in]     re row extent of B
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ops_trsm_right(const T* TDLS_RESTRICT lu, T* TDLS_RESTRICT B, const int kd, const int re) {
        for (int k = 0; k < kd; ++k) {
            const T U_kk_inv = lu[k * TS + k];
            for (int i = 0; i < re; ++i)
                B[i * TS + k] *= U_kk_inv;
            for (int j = k + 1; j < kd; ++j) {
                const T U_kj = lu[k * TS + j];
                for (int i = 0; i < re; ++i)
                    B[i * TS + j] -= B[i * TS + k] * U_kj;
            }
        }
    }

    /// \brief Ct -= At*Bt with per-element dot-product accumulation.
    /// At is re x kd, Bt is kd x ce, Ct is re x ce.
    /// \param[in,out] Ct accumulator tile
    /// \param[in]     At left factor tile
    /// \param[in]     Bt right factor tile
    /// \param[in]     re row extent of Ct and At
    /// \param[in]     ce column extent of Ct and Bt
    /// \param[in]     kd inner extent (columns of At, rows of Bt)
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ops_gemm_sub(T* TDLS_RESTRICT Ct, const T* TDLS_RESTRICT At, const T* TDLS_RESTRICT Bt,
                 const int re, const int ce, const int kd) {
        for (int i = 0; i < re; ++i) {
            for (int j = 0; j < ce; ++j) {
                T sum = T(0);
                for (int k = 0; k < kd; ++k)
                    sum += At[i * TS + k] * Bt[k * TS + j];
                Ct[i * TS + j] -= sum;
            }
        }
    }

    /* =====================================================================
       Remote <-> register tile movement, runtime extents.
       ===================================================================== */

    /// \brief Load a re x ce tile through a cached physical-row segment.
    /// \param[in]  n        system dimension
    /// \param[in]  A        matrix (caller-pre-offset)
    /// \param[in]  A_stride element stride of A
    /// \param[in]  prow     physical rows of the tile
    /// \param[in]  col0     first global column of the tile
    /// \param[out] t        destination register tile
    /// \param[in]  re       tile row extent
    /// \param[in]  ce       tile column extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void load_tile(const int n, const T* TDLS_RESTRICT A,
                                                            const int A_stride,
                                                            const int* TDLS_RESTRICT prow,
                                                            const int col0, T* TDLS_RESTRICT t,
                                                            const int re, const int ce) {
        for (int i = 0; i < re; ++i) {
            for (int j = 0; j < ce; ++j)
                t[i * TS + j] = TDLS_LUPP_DYN_A(prow[i], col0 + j);
        }
    }

    /// \brief Store a re x ce tile through a cached physical-row segment.
    /// \param[in]     n        system dimension
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in]     prow     physical rows of the tile
    /// \param[in]     col0     first global column of the tile
    /// \param[in]     t        source register tile
    /// \param[in]     re       tile row extent
    /// \param[in]     ce       tile column extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    store_tile(const int n, T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT prow,
               const int col0, const T* TDLS_RESTRICT t, const int re, const int ce) {
        for (int i = 0; i < re; ++i) {
            for (int j = 0; j < ce; ++j)
                TDLS_LUPP_DYN_A(prow[i], col0 + j) = t[i * TS + j];
        }
    }

    /// \brief Load a re x ce tile, reading the permutation inline (one
    /// read per row).
    /// \param[in]  n          system dimension
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    /// \param[in]  re         tile row extent
    /// \param[in]  ce         tile column extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                  const int* TDLS_RESTRICT piv, const int piv_stride, const int row0,
                  const int col0, T* TDLS_RESTRICT t, const int re, const int ce) {
        for (int i = 0; i < re; ++i) {
            const int phys = TDLS_LUPP_DYN_PIV(row0 + i);
            for (int j = 0; j < ce; ++j)
                t[i * TS + j] = TDLS_LUPP_DYN_A(phys, col0 + j);
        }
    }

    /// \brief Store a re x ce tile, reading the permutation inline (one
    /// read per row).
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     row0       first global (logical) row of the tile
    /// \param[in]     col0       first global column of the tile
    /// \param[in]     t          source register tile
    /// \param[in]     re         tile row extent
    /// \param[in]     ce         tile column extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    store_tile_piv(const int n, T* TDLS_RESTRICT A, const int A_stride,
                   const int* TDLS_RESTRICT piv, const int piv_stride, const int row0,
                   const int col0, const T* TDLS_RESTRICT t, const int re, const int ce) {
        for (int i = 0; i < re; ++i) {
            const int phys = TDLS_LUPP_DYN_PIV(row0 + i);
            for (int j = 0; j < ce; ++j)
                TDLS_LUPP_DYN_A(phys, col0 + j) = t[i * TS + j];
        }
    }

    /// \brief Triangular variant of the diagonal-tile load for the forward
    /// substitution: only the strict lower triangle of L\\U is read.
    /// \param[in]  n          system dimension
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    /// \param[in]  re         tile extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv_lower(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                        const int* TDLS_RESTRICT piv, const int piv_stride, const int row0,
                        const int col0, T* TDLS_RESTRICT t, const int re) {
        for (int i = 1; i < re; ++i) {
            const int phys = TDLS_LUPP_DYN_PIV(row0 + i);
            for (int j = 0; j < i; ++j)
                t[i * TS + j] = TDLS_LUPP_DYN_A(phys, col0 + j);
        }
    }

    /// \brief Triangular variant of the diagonal-tile load for the
    /// backward substitution: only the upper triangle including the
    /// diagonal is read.
    /// \param[in]  n          system dimension
    /// \param[in]  A          matrix (caller-pre-offset)
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        permutation (logical -> physical row)
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  row0       first global (logical) row of the tile
    /// \param[in]  col0       first global column of the tile
    /// \param[out] t          destination register tile
    /// \param[in]  re         tile extent
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    load_tile_piv_upper(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                        const int* TDLS_RESTRICT piv, const int piv_stride, const int row0,
                        const int col0, T* TDLS_RESTRICT t, const int re) {
        for (int i = 0; i < re; ++i) {
            const int phys = TDLS_LUPP_DYN_PIV(row0 + i);
            for (int j = i; j < re; ++j)
                t[i * TS + j] = TDLS_LUPP_DYN_A(phys, col0 + j);
        }
    }

    /* =====================================================================
       Diagonal-tile factorization with out-of-tile pivoting. Same
       algorithm as the static TiledLUpp solver, runtime column/extent parameters,
       direct (unpredicated) fused-RHS swap.
       ===================================================================== */

    /// \brief One column step of the diagonal-tile factorization: pivot
    /// search (in-tile, then out-of-tile recovery), permutation update,
    /// row swap or cross-tile pull, column elimination.
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \tparam fuse_rhs apply the pivot swaps to the fused RHS y
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     k0         first global row/column of the tile
    /// \param[in,out] tile       register-resident diagonal tile
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in]     c          tile column to factor
    /// \param[in]     ke         extent of the diagonal tile
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    /// \return false when the matrix is singular at this column.
    template<bool oot_diag, bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factor_diag_column(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                       const int piv_stride, const int k0, T* TDLS_RESTRICT tile, int& oot_count,
                       const int c, const int ke, T* TDLS_RESTRICT y, const int rhs_stride) {

        const int gc = k0 + c; // global column

        // In-tile pivot search (rows c..ke of the register tile)
        int best_r = c;
        T best     = std::fabs(tile[c * TS + c]);
        for (int r = c + 1; r < ke; ++r) {
            const T v = std::fabs(tile[r * TS + c]);
            if (v > best) {
                best   = v;
                best_r = r;
            }
        }

        int piv_row; // winning global (logical) row

        if (best >= TiledLUppSolverConfig::oot_threshold) {
            piv_row = k0 + best_r;
        } else if (ke < TS) {
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
            if constexpr (oot_diag) ++oot_count;
            T gbest       = best;
            int gbest_row = k0 + best_r;

            for (int row = k0 + ke; row < n; ++row) {
                const int phys = TDLS_LUPP_DYN_PIV(row);

                T corrected = TDLS_LUPP_DYN_A(phys, gc);
                if constexpr (Sched == TiledLUppSchedule::LeftLooking) {
                    for (int bj0 = 0; bj0 < k0; bj0 += TS)
                        for (int p = 0; p < TS; ++p)
                            corrected -= TDLS_LUPP_DYN_A(phys, bj0 + p) *
                                         TDLS_LUPP_DYN_A(TDLS_LUPP_DYN_PIV(bj0 + p), gc);
                }

                if (c > 0) {
                    T L_row[TS];
                    for (int t = 0; t < c; ++t) {
                        T a_t = TDLS_LUPP_DYN_A(phys, k0 + t);
                        if constexpr (Sched == TiledLUppSchedule::LeftLooking) {
                            for (int bj0 = 0; bj0 < k0; bj0 += TS)
                                for (int p = 0; p < TS; ++p)
                                    a_t -= TDLS_LUPP_DYN_A(phys, bj0 + p) *
                                           TDLS_LUPP_DYN_A(TDLS_LUPP_DYN_PIV(bj0 + p), k0 + t);
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
                // in-tile pivot, so stop scanning.
                if constexpr (TiledLUppSolverConfig::oot_first_acceptable)
                    if (v >= TiledLUppSolverConfig::oot_threshold) break;
            }

            if (gbest < TiledLUppSolverConfig::singular_eps) return false;
            piv_row = gbest_row;
        }

        {
            // Swap the permutation entries unconditionally (self-swap is a
            // bitwise no-op, and the branch would diverge on GPU).
            const int tmp              = TDLS_LUPP_DYN_PIV(gc);
            TDLS_LUPP_DYN_PIV(gc)      = TDLS_LUPP_DYN_PIV(piv_row);
            TDLS_LUPP_DYN_PIV(piv_row) = tmp;

            if constexpr (fuse_rhs) {
                // The fused RHS follows the rows through pivoting.
                const T ty               = TDLS_LUPP_DYN_Y(gc);
                TDLS_LUPP_DYN_Y(gc)      = TDLS_LUPP_DYN_Y(piv_row);
                TDLS_LUPP_DYN_Y(piv_row) = ty;
            }

            if (piv_row < k0 + ke) {
                ops_swap_rows(tile, c, piv_row - k0, ke);
            } else {
                // Cross-tile swap: pull the new row into the tile and
                // replay everything it missed - prior tiles (LL), then
                // the current tile's factored columns, on the FULL row.
                const int phys = TDLS_LUPP_DYN_PIV(gc);
                for (int j = 0; j < ke; ++j) {
                    tile[c * TS + j] = TDLS_LUPP_DYN_A(phys, k0 + j);
                    if constexpr (Sched == TiledLUppSchedule::LeftLooking) {
                        for (int bj0 = 0; bj0 < k0; bj0 += TS)
                            for (int p = 0; p < TS; ++p)
                                tile[c * TS + j] -=
                                    TDLS_LUPP_DYN_A(phys, bj0 + p) *
                                    TDLS_LUPP_DYN_A(TDLS_LUPP_DYN_PIV(bj0 + p), k0 + j);
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
                    for (int j = c; j < ke; ++j) {
                        for (int t = 0; t < c; ++t)
                            tile[c * TS + j] -= L_row[t] * tile[t * TS + j];
                    }
                }
            }
        }

        ops_eliminate_column(tile, c, ke, ke);
        return true;
    }

    /// \brief Factor the ke x ke diagonal tile in registers, with
    /// out-of-tile pivot recovery (drives the per-column loop of
    /// factor_diag_column).
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \tparam fuse_rhs apply the pivot swaps to the fused RHS y
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     k0         first global row/column of the tile
    /// \param[in,out] tile       register-resident diagonal tile, loaded
    ///                (and, LL: prior-corrected) by the caller
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in]     ke         extent of the diagonal tile
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    /// \return false on a singular matrix.
    template<bool oot_diag, bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factor_diag_tile(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                     const int piv_stride, const int k0, T* TDLS_RESTRICT tile, int& oot_count,
                     const int ke, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        for (int c = 0; c < ke; ++c) {
            if (!factor_diag_column<oot_diag, fuse_rhs>(n, A, A_stride, piv, piv_stride, k0, tile,
                                                        oot_count, c, ke, y, rhs_stride))
                return false;
        }
        return true;
    }

    /* =====================================================================
       RIGHT-LOOKING schedule.
       ===================================================================== */

    /// \brief RL: one row-panel tile update, Akj := L^-1 Akj.
    /// \param[in]     n        system dimension
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in]     pk       physical rows of the diagonal block row
    /// \param[in]     tile     factored diagonal tile
    /// \param[in]     j0       first global column of the updated tile
    /// \param[in]     ke       extent of the factored diagonal tile
    /// \param[in]     je       column extent of the updated tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_trsm_right_one(const int n, T* TDLS_RESTRICT A, const int A_stride,
                      const int* TDLS_RESTRICT pk, const T* TDLS_RESTRICT tile, const int j0,
                      const int ke, const int je) {
        T Akj[TS * TS];
        load_tile(n, A, A_stride, pk, j0, Akj, ke, je);
        ops_trsm_left_unit(tile, Akj, ke, je);
        store_tile(n, A, A_stride, pk, j0, Akj, ke, je);
    }

    /// \brief RL: one Schur-complement tile update, Aij -= Aik * Akj,
    /// streaming the factored Akj row by row from remote memory.
    /// \param[in]     n        system dimension
    /// \param[in,out] A        matrix (caller-pre-offset)
    /// \param[in]     A_stride element stride of A
    /// \param[in]     pk       physical rows of the factored block row
    /// \param[in]     pi       physical rows of the updated block row
    /// \param[in]     Aik      register L panel of the updated block row
    /// \param[in]     j0       first global column of the updated tile
    /// \param[in]     ke       inner extent (current step)
    /// \param[in]     ie       row extent of the updated tile
    /// \param[in]     je       column extent of the updated tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_schur_one(const int n, T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT pk,
                 const int* TDLS_RESTRICT pi, const T* TDLS_RESTRICT Aik, const int j0,
                 const int ke, const int ie, const int je) {
        T Aij[TS * TS];
        load_tile(n, A, A_stride, pi, j0, Aij, ie, je);

        for (int p = 0; p < ke; ++p) {
            T Akj_row[TS];
            for (int j = 0; j < je; ++j)
                Akj_row[j] = TDLS_LUPP_DYN_A(pk[p], j0 + j);
            for (int i = 0; i < ie; ++i) {
                const T L_ip = Aik[i * TS + p];
                for (int j = 0; j < je; ++j)
                    Aij[i * TS + j] -= L_ip * Akj_row[j];
            }
        }

        store_tile(n, A, A_stride, pi, j0, Aij, ie, je);
    }

    /// \brief RL: TRSM down + Schur sweep of one row block below the
    /// diagonal, with the optional fused forward-substitution push.
    /// \tparam fuse_rhs push the fused RHS y along (solve_fused)
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     pk         physical rows of the diagonal tile
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in]     i0         first global row of the updated block row
    /// \param[in]     ke         extent of the diagonal tile
    /// \param[in]     ie         row extent of the updated block row
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    rl_update_row_one(const int n, T* TDLS_RESTRICT A, const int A_stride,
                      const int* TDLS_RESTRICT piv, const int piv_stride,
                      const int* TDLS_RESTRICT pk, const T* TDLS_RESTRICT tile, const int k,
                      const int i0, const int ke, const int ie, T* TDLS_RESTRICT y = nullptr,
                      const int rhs_stride = 1) {
        const int k0 = k * TS;
        const int nt = num_tiles(n);

        int pi[TS];
        for (int i = 0; i < ie; ++i)
            pi[i] = TDLS_LUPP_DYN_PIV(i0 + i);

        // TRSM down: Aik := Aik * U^-1
        T Aik[TS * TS];
        load_tile(n, A, A_stride, pi, k0, Aik, ie, ke);
        ops_trsm_right(tile, Aik, ke, ie);
        store_tile(n, A, A_stride, pi, k0, Aik, ie, ke);

        // Fused forward substitution: push the solved y_k segment into
        // this row block while its L panel sits in registers.
        if constexpr (fuse_rhs) {
            for (int r = 0; r < ie; ++r) {
                T sum = T(0);
                for (int j = 0; j < ke; ++j)
                    sum += Aik[r * TS + j] * TDLS_LUPP_DYN_Y(k0 + j);
                TDLS_LUPP_DYN_Y(i0 + r) -= sum;
            }
        }

        // Schur sweep over the trailing columns
        for (int tj = k + 1; tj < nt; ++tj)
            rl_schur_one(n, A, A_stride, pk, pi, Aik, tj * TS, ke, ie, tile_extent(tj * TS, n));
    }

    /// \brief RL: one full factorization step (diagonal tile + trailing
    /// updates).
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \tparam fuse_rhs apply the step to the fused RHS y (solve_fused)
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    /// \return false on a singular matrix.
    template<bool oot_diag, bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    rl_step(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
            const int piv_stride, const int k, int& oot_count, T* TDLS_RESTRICT y = nullptr,
            const int rhs_stride = 1) {
        const int k0 = k * TS;
        const int nt = num_tiles(n);
        const int ke = tile_extent(k0, n);

        T tile[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, k0, k0, tile, ke, ke);

        if (!factor_diag_tile<oot_diag, fuse_rhs>(n, A, A_stride, piv, piv_stride, k0, tile,
                                                  oot_count, ke, y, rhs_stride))
            return false;

        // Physical rows of the tile after the swaps of this step
        int pk[TS];
        for (int i = 0; i < ke; ++i)
            pk[i] = TDLS_LUPP_DYN_PIV(k0 + i);

        store_tile(n, A, A_stride, pk, k0, tile, ke, ke);

        // Fused forward substitution: this tile's y segment is final from
        // here on - unit-lower-solve it while the tile is in registers.
        if constexpr (fuse_rhs) {
            for (int kk = 0; kk < ke; ++kk) {
                for (int i = kk + 1; i < ke; ++i)
                    TDLS_LUPP_DYN_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_DYN_Y(k0 + kk);
            }
        }

        // TRSM right over the row panel
        for (int tj = k + 1; tj < nt; ++tj)
            rl_trsm_right_one(n, A, A_stride, pk, tile, tj * TS, ke, tile_extent(tj * TS, n));

        // TRSM down + Schur over the rows below
        for (int ti = k + 1; ti < nt; ++ti)
            rl_update_row_one<fuse_rhs>(n, A, A_stride, piv, piv_stride, pk, tile, k, ti * TS, ke,
                                        tile_extent(ti * TS, n), y, rhs_stride);
        return true;
    }

    /* =====================================================================
       LEFT-LOOKING schedule.
       ===================================================================== */

    /// \brief LL: t (re x ce, rows row0.., cols col0..) -= sum over prior
    /// tiles bj < k0/TS of L(row0.., bj) * U(bj, col0..). All prior tiles
    /// are full.
    /// \param[in]     n          system dimension
    /// \param[in]     A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     row0       first global row of the corrected tile
    /// \param[in]     col0       first global column of the corrected tile
    /// \param[in]     k0         first row/column of the current step
    /// \param[in,out] t          register tile being corrected
    /// \param[in]     re         row extent of the corrected tile
    /// \param[in]     ce         column extent of the corrected tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_correct_tile(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                    const int* TDLS_RESTRICT piv, const int piv_stride, const int row0,
                    const int col0, const int k0, T* TDLS_RESTRICT t, const int re, const int ce) {
        for (int bj0 = 0; bj0 < k0; bj0 += TS) {
            T Lt[TS * TS];
            load_tile_piv(n, A, A_stride, piv, piv_stride, row0, bj0, Lt, re, TS);
            T Ut[TS * TS];
            load_tile_piv(n, A, A_stride, piv, piv_stride, bj0, col0, Ut, TS, ce);
            ops_gemm_sub(t, Lt, Ut, re, ce, TS);
        }
    }

    /// \brief LL: correct + TRSM one L-panel tile below the diagonal, with
    /// the optional fused forward-substitution push.
    /// \tparam fuse_rhs push the fused RHS y along (solve_fused)
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k0         first global row/column of the step
    /// \param[in]     i0         first global row of the updated tile
    /// \param[in]     ke         extent of the diagonal tile
    /// \param[in]     ie         row extent of the updated tile
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    template<bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_update_below_one(const int n, T* TDLS_RESTRICT A, const int A_stride,
                        const int* TDLS_RESTRICT piv, const int piv_stride,
                        const T* TDLS_RESTRICT tile, const int k0, const int i0, const int ke,
                        const int ie, T* TDLS_RESTRICT y = nullptr, const int rhs_stride = 1) {
        T B[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, i0, k0, B, ie, ke);
        ll_correct_tile(n, A, A_stride, piv, piv_stride, i0, k0, k0, B, ie, ke);
        ops_trsm_right(tile, B, ke, ie);
        store_tile_piv(n, A, A_stride, piv, piv_stride, i0, k0, B, ie, ke);

        // Fused forward substitution: B is the final L(i,k) panel - push
        // the solved y_k segment into this row block.
        if constexpr (fuse_rhs) {
            for (int r = 0; r < ie; ++r) {
                T sum = T(0);
                for (int j = 0; j < ke; ++j)
                    sum += B[r * TS + j] * TDLS_LUPP_DYN_Y(k0 + j);
                TDLS_LUPP_DYN_Y(i0 + r) -= sum;
            }
        }
    }

    /// \brief LL: correct + TRSM one U-panel tile right of the diagonal.
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     tile       factored diagonal tile
    /// \param[in]     k0         first global row/column of the step
    /// \param[in]     j0         first global column of the updated tile
    /// \param[in]     ke         extent of the diagonal tile
    /// \param[in]     je         column extent of the updated tile
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    ll_update_right_one(const int n, T* TDLS_RESTRICT A, const int A_stride,
                        const int* TDLS_RESTRICT piv, const int piv_stride,
                        const T* TDLS_RESTRICT tile, const int k0, const int j0, const int ke,
                        const int je) {
        T B[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, k0, j0, B, ke, je);
        ll_correct_tile(n, A, A_stride, piv, piv_stride, k0, j0, k0, B, ke, je);
        ops_trsm_left_unit(tile, B, ke, je);
        store_tile_piv(n, A, A_stride, piv, piv_stride, k0, j0, B, ke, je);
    }

    /// \brief LL: one full factorization step (correct + factor the
    /// diagonal tile, then its L and U panels).
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \tparam fuse_rhs apply the step to the fused RHS y (solve_fused)
    /// \param[in]     n          system dimension
    /// \param[in,out] A          matrix (caller-pre-offset)
    /// \param[in]     A_stride   element stride of A
    /// \param[in,out] piv        permutation (logical -> physical row)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     k          step index (k0 = k*TS)
    /// \param[in,out] oot_count  out-of-tile search counter (oot_diag)
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    /// \return false on a singular matrix.
    template<bool oot_diag, bool fuse_rhs>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    ll_step(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
            const int piv_stride, const int k, int& oot_count, T* TDLS_RESTRICT y = nullptr,
            const int rhs_stride = 1) {
        const int k0 = k * TS;
        const int nt = num_tiles(n);
        const int ke = tile_extent(k0, n);

        T tile[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, k0, k0, tile, ke, ke);
        ll_correct_tile(n, A, A_stride, piv, piv_stride, k0, k0, k0, tile, ke, ke);

        if (!factor_diag_tile<oot_diag, fuse_rhs>(n, A, A_stride, piv, piv_stride, k0, tile,
                                                  oot_count, ke, y, rhs_stride))
            return false;

        store_tile_piv(n, A, A_stride, piv, piv_stride, k0, k0, tile, ke, ke);

        // Fused forward substitution: this tile's y segment is final from
        // here on - unit-lower-solve it while the tile is in registers.
        if constexpr (fuse_rhs) {
            for (int kk = 0; kk < ke; ++kk) {
                for (int i = kk + 1; i < ke; ++i)
                    TDLS_LUPP_DYN_Y(k0 + i) -= tile[i * TS + kk] * TDLS_LUPP_DYN_Y(k0 + kk);
            }
        }

        // L panel below the diagonal
        for (int ti = k + 1; ti < nt; ++ti)
            ll_update_below_one<fuse_rhs>(n, A, A_stride, piv, piv_stride, tile, k0, ti * TS, ke,
                                          tile_extent(ti * TS, n), y, rhs_stride);

        // U row panel right of the diagonal
        for (int tj = k + 1; tj < nt; ++tj)
            ll_update_right_one(n, A, A_stride, piv, piv_stride, tile, k0, tj * TS, ke,
                                tile_extent(tj * TS, n));
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
    /// \tparam oot_diag when false, the out-of-tile diagnostics are
    ///         compiled out entirely - use the overload without the
    ///         out-parameter
    /// \tparam fuse_rhs internal hook of solve_fused: folds the forward
    ///         substitution of y into the factorization
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          matrix, pre-offset by the caller
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \param[in,out] y          fused right-hand side (fuse_rhs only)
    /// \param[in]     rhs_stride element stride of y
    /// \return false on a singular matrix.
    template<bool oot_diag = true, bool fuse_rhs = false>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    factorize(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
              const int piv_stride, int& oot_count, T* TDLS_RESTRICT y = nullptr,
              const int rhs_stride = 1) {

        if constexpr (oot_diag) oot_count = 0;

        for (int i = 0; i < n; ++i)
            TDLS_LUPP_DYN_PIV(i) = i;

        const int nt = num_tiles(n);
        for (int k = 0; k < nt; ++k) {
            if constexpr (Sched == TiledLUppSchedule::RightLooking) {
                if (!rl_step<oot_diag, fuse_rhs>(n, A, A_stride, piv, piv_stride, k, oot_count, y,
                                                 rhs_stride))
                    return false;
            } else {
                if (!ll_step<oot_diag, fuse_rhs>(n, A, A_stride, piv, piv_stride, k, oot_count, y,
                                                 rhs_stride))
                    return false;
            }
        }

        return true;
    }

    /// \brief Diagnostics-free factorize overload: no out-of-tile
    /// out-parameter at all.
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          matrix, pre-offset by the caller
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \return false on a singular matrix.
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool factorize(const int n, T* TDLS_RESTRICT A,
                                                            const int A_stride,
                                                            int* TDLS_RESTRICT piv,
                                                            const int piv_stride) {
        int unused = 0;
        return factorize<false>(n, A, A_stride, piv, piv_stride, unused);
    }

    /* =====================================================================
       SUBSTITUTION - schedule-independent.
       ===================================================================== */

    /// \brief Forward push: subtract L(m,k) * x_k from the x_m segment,
    /// for W columns at once.
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k0          first global row of the solved segment
    /// \param[in]     m0          first global row of the target segment
    /// \param[in]     ke          extent of the solved segment's tile
    /// \param[in]     me          extent of the target segment's tile
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_push_one(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                 const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                 const int rhs_stride, const int xcol_stride, const int k0, const int m0,
                 const int ke, const int me) {
        T Lmk[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, m0, k0, Lmk, me, ke);
        for (int i = 0; i < me; ++i) {
            for (int w = 0; w < W; ++w) {
                T sum = T(0);
                for (int j = 0; j < ke; ++j)
                    sum += Lmk[i * TS + j] * TDLS_LUPP_DYN_XW(w, k0 + j);
                TDLS_LUPP_DYN_XW(w, m0 + i) -= sum;
            }
        }
    }

    /// \brief Forward step: unit-lower solve the diagonal tile's segment,
    /// then push it into the tiles below.
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k           step index (k0 = k*TS)
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_step(const int n, const T* TDLS_RESTRICT A, const int A_stride,
             const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
             const int rhs_stride, const int xcol_stride, const int k) {
        const int k0 = k * TS;
        const int nt = num_tiles(n);
        const int ke = tile_extent(k0, n);

        T Lkk[TS * TS];
        load_tile_piv_lower(n, A, A_stride, piv, piv_stride, k0, k0, Lkk, ke);

        // In-tile unit-lower solve
        for (int kk = 0; kk < ke; ++kk) {
            for (int i = kk + 1; i < ke; ++i) {
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_DYN_XW(w, k0 + i) -= Lkk[i * TS + kk] * TDLS_LUPP_DYN_XW(w, k0 + kk);
            }
        }

        // Push into the tiles below
        for (int m = k + 1; m < nt; ++m)
            fwd_push_one<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, m * TS,
                            ke, tile_extent(m * TS, n));
    }

    /// \brief Backward pull: subtract U(k,m) * x_m from the x_k segment,
    /// for W columns at once.
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k0          first global row of the updated segment
    /// \param[in]     m0          first global row of the trailing segment
    /// \param[in]     ke          extent of the updated segment's tile
    /// \param[in]     me          extent of the trailing segment's tile
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_pull_one(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                 const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                 const int rhs_stride, const int xcol_stride, const int k0, const int m0,
                 const int ke, const int me) {
        T Ukm[TS * TS];
        load_tile_piv(n, A, A_stride, piv, piv_stride, k0, m0, Ukm, ke, me);
        for (int i = 0; i < ke; ++i) {
            for (int w = 0; w < W; ++w) {
                T sum = T(0);
                for (int j = 0; j < me; ++j)
                    sum += Ukm[i * TS + j] * TDLS_LUPP_DYN_XW(w, m0 + j);
                TDLS_LUPP_DYN_XW(w, k0 + i) -= sum;
            }
        }
    }

    /// \brief Backward step: pull the trailing contributions, then
    /// upper-solve the diagonal tile's segment.
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    /// \param[in]     k           step index (k0 = k*TS)
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_step(const int n, const T* TDLS_RESTRICT A, const int A_stride,
             const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
             const int rhs_stride, const int xcol_stride, const int k) {
        const int k0 = k * TS;
        const int nt = num_tiles(n);
        const int ke = tile_extent(k0, n);

        // Pull the trailing contributions
        for (int m = k + 1; m < nt; ++m)
            bwd_pull_one<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k0, m * TS,
                            ke, tile_extent(m * TS, n));

        // In-tile upper solve
        T Ukk[TS * TS];
        load_tile_piv_upper(n, A, A_stride, piv, piv_stride, k0, k0, Ukk, ke);

        for (int kk = ke - 1; kk >= 0; --kk) {
            for (int w = 0; w < W; ++w)
                TDLS_LUPP_DYN_XW(w, k0 + kk) *= Ukk[kk * TS + kk]; // diag holds 1/pivot
            for (int i = 0; i < kk; ++i) {
                for (int w = 0; w < W; ++w)
                    TDLS_LUPP_DYN_XW(w, k0 + i) -= Ukk[i * TS + kk] * TDLS_LUPP_DYN_XW(w, k0 + kk);
            }
        }
    }

    /// \brief Backward pass alone - used by fwd_bwd and by solve_fused
    /// (whose forward pass happens inside the factorization).
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    bwd_only(const int n, const T* TDLS_RESTRICT A, const int A_stride,
             const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
             const int rhs_stride, const int xcol_stride) {
        for (int k = num_tiles(n) - 1; k >= 0; --k)
            bwd_step<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k);
    }

    /// \brief Triangular solves on already-permuted column(s) x. W columns
    /// are processed per tile visit, so every L/U tile is loaded once for
    /// the whole block instead of once per column (W=1 = the single-RHS
    /// case).
    /// \tparam W number of columns processed together
    /// \param[in]     n           system dimension
    /// \param[in]     A           factored matrix (caller-pre-offset)
    /// \param[in]     A_stride    element stride of A
    /// \param[in]     piv         permutation from factorize
    /// \param[in]     piv_stride  element stride of piv
    /// \param[in,out] x           solution column(s)
    /// \param[in]     rhs_stride  element stride of x
    /// \param[in]     xcol_stride element stride between columns of x
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    fwd_bwd(const int n, const T* TDLS_RESTRICT A, const int A_stride, const int* TDLS_RESTRICT piv,
            const int piv_stride, T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride) {
        const int nt = num_tiles(n);
        for (int k = 0; k < nt; ++k)
            fwd_step<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride, k);

        bwd_only<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride);
    }

    /* =====================================================================
       SUBSTITUTION - public entry points.
       ===================================================================== */

    /// \brief Solve x := U^-1 L^-1 P b from a prior factorize. b and x
    /// must not alias (use substitute_inplace for the aliased
    /// single-buffer pattern).
    /// \param[in]  n          system dimension
    /// \param[in]  A          factored matrix produced by factorize
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        permutation produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  b          right-hand side, in original (unpermuted) order
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of b and x
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute(const int n, const T* TDLS_RESTRICT A, const int A_stride,
               const int* TDLS_RESTRICT piv, const int piv_stride, const T* TDLS_RESTRICT b,
               T* TDLS_RESTRICT x, const int rhs_stride) {
        for (int i = 0; i < n; ++i)
            TDLS_LUPP_DYN_X(i) = TDLS_LUPP_DYN_B(TDLS_LUPP_DYN_PIV(i));
        fwd_bwd<1>(n, A, A_stride, piv, piv_stride, x, rhs_stride, 0);
    }

    /// \brief Solve with b = e_col generated on the fly - the
    /// consistent-tangent-operator path.
    ///
    /// Thin alias of the W-column variant below (W=1 collapses to exactly
    /// the single-column code, so there is no separate implementation to
    /// maintain).
    /// \param[in]  n          system dimension
    /// \param[in]  A          factored matrix produced by factorize
    /// \param[in]  A_stride   element stride of A
    /// \param[in]  piv        permutation produced by factorize
    /// \param[in]  piv_stride element stride of piv
    /// \param[in]  col        index of the canonical column e_col
    /// \param[out] x          solution
    /// \param[in]  rhs_stride element stride of x
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_canonical(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                         const int* TDLS_RESTRICT piv, const int piv_stride, const int col,
                         T* TDLS_RESTRICT x, const int rhs_stride) {
        substitute_canonical_block<1>(n, A, A_stride, piv, piv_stride, col, x, rhs_stride, 0);
    }

    /// \brief W canonical columns e_col0..e_{col0+W-1} solved per tile
    /// visit: every L/U tile is loaded once for the block instead of once
    /// per column.
    /// \tparam W number of consecutive canonical columns solved together
    /// \param[in]  n           system dimension
    /// \param[in]  A           factored matrix produced by factorize
    /// \param[in]  A_stride    element stride of A
    /// \param[in]  piv         permutation produced by factorize
    /// \param[in]  piv_stride  element stride of piv
    /// \param[in]  col0        index of the first canonical column
    /// \param[out] x           W solution columns
    /// \param[in]  rhs_stride  element stride of x
    /// \param[in]  xcol_stride element stride between columns of x
    template<int W>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_canonical_block(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                               const int* TDLS_RESTRICT piv, const int piv_stride, const int col0,
                               T* TDLS_RESTRICT x, const int rhs_stride, const int xcol_stride) {
        for (int i = 0; i < n; ++i) {
            const int p = TDLS_LUPP_DYN_PIV(i);
            for (int w = 0; w < W; ++w)
                TDLS_LUPP_DYN_XW(w, i) = (p == col0 + w) ? T(1) : T(0);
        }
        fwd_bwd<W>(n, A, A_stride, piv, piv_stride, x, rhs_stride, xcol_stride);
    }

    /// \brief Solve in place: x already holds the unpermuted RHS on entry
    /// and the solution on exit.
    ///
    /// The permutation is applied in place by cycle decomposition, each
    /// cycle rotated once from its smallest index. Up to n = 64 a 64-bit
    /// bitmask tracks the visited rows; beyond, a cycle-leader scan is
    /// used instead (zero storage, an extra integer walk of each orbit).
    /// The dispatch is warp-uniform on GPU (n is the same for every lane).
    /// Both paths move the same values in the same order.
    /// \param[in]     n          system dimension
    /// \param[in]     A          factored matrix produced by factorize
    /// \param[in]     A_stride   element stride of A
    /// \param[in]     piv        permutation produced by factorize
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] x          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of x
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    substitute_inplace(const int n, const T* TDLS_RESTRICT A, const int A_stride,
                       const int* TDLS_RESTRICT piv, const int piv_stride, T* TDLS_RESTRICT x,
                       const int rhs_stride) {
        if (n <= 64) {
            // Bitmask cycle decomposition: one visited bit per row, the
            // whole state in a single 64-bit register.
            unsigned long long visited = 0ull;
            for (int s = 0; s < n; ++s) {
                if ((visited >> s) & 1ull) continue;
                const T tmp = TDLS_LUPP_DYN_X(s);
                int cur     = s;
                int nxt     = TDLS_LUPP_DYN_PIV(cur);
                while (nxt != s) {
                    TDLS_LUPP_DYN_X(cur) = TDLS_LUPP_DYN_X(nxt);
                    visited |= 1ull << cur;
                    cur = nxt;
                    nxt = TDLS_LUPP_DYN_PIV(cur);
                }
                TDLS_LUPP_DYN_X(cur) = tmp;
                visited |= 1ull << cur;
            }
        } else {
            // Cycle-leader scan (no visited storage): a cycle is rotated
            // only when reached from its smallest index, detected by
            // walking the orbit. Values and order match the mask path.
            for (int s = 0; s < n; ++s) {
                int probe = TDLS_LUPP_DYN_PIV(s);
                while (probe > s)
                    probe = TDLS_LUPP_DYN_PIV(probe);
                if (probe != s) continue;

                const T tmp = TDLS_LUPP_DYN_X(s);
                int cur     = s;
                int nxt     = TDLS_LUPP_DYN_PIV(cur);
                while (nxt != s) {
                    TDLS_LUPP_DYN_X(cur) = TDLS_LUPP_DYN_X(nxt);
                    cur                  = nxt;
                    nxt                  = TDLS_LUPP_DYN_PIV(cur);
                }
                TDLS_LUPP_DYN_X(cur) = tmp;
            }
        }
        fwd_bwd<1>(n, A, A_stride, piv, piv_stride, x, rhs_stride, 0);
    }

    /* =====================================================================
       SOLVE - convenience wrappers.
       ===================================================================== */

    /// \brief factorize + substitute in one call.
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization (usable for
    ///                further substitute* calls)
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \return false on a singular matrix.
    template<bool oot_diag = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
          const int piv_stride, const T* TDLS_RESTRICT b, T* TDLS_RESTRICT x, const int rhs_stride,
          int& oot_count) {
        if (!factorize<oot_diag>(n, A, A_stride, piv, piv_stride, oot_count)) return false;
        substitute(n, A, A_stride, piv, piv_stride, b, x, rhs_stride);
        return true;
    }

    /// \brief Diagnostics-free solve overload: no out-of-tile
    /// out-parameter at all.
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in]     b          right-hand side, in original order
    /// \param[out]    x          solution
    /// \param[in]     rhs_stride element stride of b and x
    /// \return false on a singular matrix.
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool solve(const int n, T* TDLS_RESTRICT A,
                                                        const int A_stride, int* TDLS_RESTRICT piv,
                                                        const int piv_stride,
                                                        const T* TDLS_RESTRICT b,
                                                        T* TDLS_RESTRICT x, const int rhs_stride) {
        int unused = 0;
        return solve<false>(n, A, A_stride, piv, piv_stride, b, x, rhs_stride, unused);
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
    /// \tparam oot_diag compile the out-of-tile counter in or out
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y
    /// \param[out]    oot_count  number of columns that needed the
    ///                out-of-tile pivot search
    /// \return false on a singular matrix (y left partially updated).
    template<bool oot_diag = true>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve_fused(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride, int& oot_count) {
        if (!factorize<oot_diag, true>(n, A, A_stride, piv, piv_stride, oot_count, y, rhs_stride))
            return false;

        // Backward pass only - the forward one happened inside factorize.
        bwd_only<1>(n, A, A_stride, piv, piv_stride, y, rhs_stride, 0);
        return true;
    }

    /// \brief Diagnostics-free solve_fused overload: no out-of-tile
    /// out-parameter at all.
    /// \param[in]     n          system dimension (n >= 2)
    /// \param[in,out] A          on entry the matrix (pre-offset by the
    ///                caller), on exit its factorization
    /// \param[in]     A_stride   element stride of A
    /// \param[out]    piv        permutation storage (always caller-provided)
    /// \param[in]     piv_stride element stride of piv
    /// \param[in,out] y          right-hand side on entry, solution on exit
    /// \param[in]     rhs_stride element stride of y
    /// \return false on a singular matrix (y left partially updated).
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static bool
    solve_fused(const int n, T* TDLS_RESTRICT A, const int A_stride, int* TDLS_RESTRICT piv,
                const int piv_stride, T* TDLS_RESTRICT y, const int rhs_stride) {
        int unused = 0;
        return solve_fused<false>(n, A, A_stride, piv, piv_stride, y, rhs_stride, unused);
    }
};



#undef TDLS_LUPP_DYN_A
#undef TDLS_LUPP_DYN_PIV
#undef TDLS_LUPP_DYN_X
#undef TDLS_LUPP_DYN_B
#undef TDLS_LUPP_DYN_XW
#undef TDLS_LUPP_DYN_Y



} // namespace tdls

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILED_LUPP_SOLVER_DYNAMIC_HPP
