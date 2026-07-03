#ifndef TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP
#define TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP



/// \file
/// \brief Register-tile micro-kernels of the TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Building blocks shared by the right- and left-looking schedules. A tile
/// is a TSxTS register array with row stride TS; partial (trailing) tiles
/// use the same storage with extent template parameters bounding every
/// loop, so phantom slots are never read or written and cost nothing.
///
/// These operate on register arrays only - remote-memory movement lives in
/// the TiledLUpp solvers (solver_static.hpp, solver_dynamic.hpp), next to the
/// addressing macros.
///
/// Every loop is subject to the `unroll_inner` knob: with it, the loops
/// must effectively unroll or the tiles are demoted to local memory on GPU
/// backends; without it, no pragma is emitted at all (see config.hpp).



#include <tdls/core/macros.hpp>



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



/// \brief TSxTS register-tile micro-kernels.
/// \tparam T            scalar type
/// \tparam TS           tile size (int, row stride of the register tiles)
/// \tparam unroll_inner unroll knob, forwarded from the TiledLUpp solver configuration
template<typename T, int TS, bool unroll_inner>
struct TiledLUppTileOps {

    /// \brief Row swap k <-> r, compile-time indexed on both sides.
    ///
    /// The equality test against every unrolled row index keeps the tile
    /// addressing static; a dynamic t[r*TS+j] would spill the tile.
    /// \param[in,out] t register tile
    /// \param[in]     k destination row (elimination column)
    /// \param[in]     r source row to swap in
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void swap_rows(T* TDLS_RESTRICT t, int k, int r) {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int row = 0; row < TS; ++row) {
                if (row == r) {
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < TS; ++j) {
                        const T tmp     = t[k * TS + j];
                        t[k * TS + j]   = t[row * TS + j];
                        t[row * TS + j] = tmp;
                    }
                }
            }
        } else {
            for (int row = 0; row < TS; ++row) {
                if (row == r) {
                    for (int j = 0; j < TS; ++j) {
                        const T tmp     = t[k * TS + j];
                        t[k * TS + j]   = t[row * TS + j];
                        t[row * TS + j] = tmp;
                    }
                }
            }
        }
    }

    /// \brief Gaussian elimination of column k inside the diagonal tile.
    ///
    /// Scales the sub-column by 1/pivot and updates the trailing block.
    /// Extents <R, C> bound the active part of the tile.
    /// The RECIPROCAL of the pivot is stored at the diagonal slot: every
    /// downstream consumer (trsm_right, backward substitution, out-of-tile
    /// replays) multiplies instead of dividing - the division is paid once
    /// here, where it already had to happen.
    /// \tparam R active row extent of the tile
    /// \tparam C active column extent of the tile
    /// \param[in,out] t register tile
    /// \param[in]     k column to eliminate
    template<int R, int C>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void eliminate_column(T* TDLS_RESTRICT t, int k) {
        const T inv_pivot = T(1) / t[k * TS + k];
        t[k * TS + k]     = inv_pivot;
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = k + 1; i < R; ++i) {
                t[i * TS + k] *= inv_pivot;
                TDLS_UNROLL_FORCE
                for (int j = k + 1; j < C; ++j)
                    t[i * TS + j] -= t[i * TS + k] * t[k * TS + j];
            }
        } else {
            for (int i = k + 1; i < R; ++i) {
                t[i * TS + k] *= inv_pivot;
                for (int j = k + 1; j < C; ++j)
                    t[i * TS + j] -= t[i * TS + k] * t[k * TS + j];
            }
        }
    }

    /// \brief B := L^-1 B, with L the unit lower part of the factored
    /// diagonal tile. L is KDxKD, B is KDxC.
    /// \tparam KD extent of the factored diagonal tile
    /// \tparam C  column extent of B
    /// \param[in]     lu factored diagonal tile (L\\U)
    /// \param[in,out] B  updated register tile
    template<int KD, int C>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void trsm_left_unit(const T* TDLS_RESTRICT lu,
                                                                 T* TDLS_RESTRICT B) {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int k = 0; k < KD; ++k) {
                TDLS_UNROLL_FORCE
                for (int i = k + 1; i < KD; ++i) {
                    const T L_ik = lu[i * TS + k];
                    TDLS_UNROLL_FORCE
                    for (int j = 0; j < C; ++j)
                        B[i * TS + j] -= L_ik * B[k * TS + j];
                }
            }
        } else {
            for (int k = 0; k < KD; ++k) {
                for (int i = k + 1; i < KD; ++i) {
                    const T L_ik = lu[i * TS + k];
                    for (int j = 0; j < C; ++j)
                        B[i * TS + j] -= L_ik * B[k * TS + j];
                }
            }
        }
    }

    /// \brief B := B U^-1, with U the upper part of the factored diagonal
    /// tile. U is KDxKD, B is RxKD.
    ///
    /// The diagonal of `lu` already holds the pivot reciprocals
    /// (see eliminate_column) - no divisions here.
    /// \tparam KD extent of the factored diagonal tile
    /// \tparam R  row extent of B
    /// \param[in]     lu factored diagonal tile (L\\U, reciprocal diagonal)
    /// \param[in,out] B  updated register tile
    template<int KD, int R>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void trsm_right(const T* TDLS_RESTRICT lu,
                                                             T* TDLS_RESTRICT B) {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int k = 0; k < KD; ++k) {
                const T U_kk_inv = lu[k * TS + k];
                TDLS_UNROLL_FORCE
                for (int i = 0; i < R; ++i)
                    B[i * TS + k] *= U_kk_inv;
                TDLS_UNROLL_FORCE
                for (int j = k + 1; j < KD; ++j) {
                    const T U_kj = lu[k * TS + j];
                    TDLS_UNROLL_FORCE
                    for (int i = 0; i < R; ++i)
                        B[i * TS + j] -= B[i * TS + k] * U_kj;
                }
            }
        } else {
            for (int k = 0; k < KD; ++k) {
                const T U_kk_inv = lu[k * TS + k];
                for (int i = 0; i < R; ++i)
                    B[i * TS + k] *= U_kk_inv;
                for (int j = k + 1; j < KD; ++j) {
                    const T U_kj = lu[k * TS + j];
                    for (int i = 0; i < R; ++i)
                        B[i * TS + j] -= B[i * TS + k] * U_kj;
                }
            }
        }
    }

    /// \brief Ct -= At*Bt with per-element dot-product accumulation.
    /// At is RxK, Bt is KxC, Ct is RxC.
    /// \tparam R row extent of Ct and At
    /// \tparam C column extent of Ct and Bt
    /// \tparam K inner extent (columns of At, rows of Bt)
    /// \param[in,out] Ct accumulator tile
    /// \param[in]     At left factor tile
    /// \param[in]     Bt right factor tile
    template<int R, int C, int K>
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static void
    gemm_sub(T* TDLS_RESTRICT Ct, const T* TDLS_RESTRICT At, const T* TDLS_RESTRICT Bt) {
        if constexpr (unroll_inner) {
            TDLS_UNROLL_FORCE
            for (int i = 0; i < R; ++i) {
                TDLS_UNROLL_FORCE
                for (int j = 0; j < C; ++j) {
                    T sum = T(0);
                    TDLS_UNROLL_FORCE
                    for (int k = 0; k < K; ++k)
                        sum += At[i * TS + k] * Bt[k * TS + j];
                    Ct[i * TS + j] -= sum;
                }
            }
        } else {
            for (int i = 0; i < R; ++i) {
                for (int j = 0; j < C; ++j) {
                    T sum = T(0);
                    for (int k = 0; k < K; ++k)
                        sum += At[i * TS + k] * Bt[k * TS + j];
                    Ct[i * TS + j] -= sum;
                }
            }
        }
    }
};



} // namespace tdls

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif



#endif // TDLS_SOLVERS_TILED_LUPP_TILE_OPS_HPP
