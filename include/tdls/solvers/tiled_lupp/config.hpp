#ifndef TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP
#define TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP



/// \file
/// \brief Compile-time configuration of the TiledLUpp solver family.
/// \author Tristan Chenaille
///
/// Every knob is a `static constexpr` member of a config type passed as the
/// `TiledLUppSolverConfig` template argument of the TiledLUpp solvers. TiledLUppDefaultConfig holds the
/// tuned defaults; a caller overrides them by providing its own type with
/// the same members.



#include <limits>
#include <type_traits>

#include <tdls/core/macros.hpp>



namespace tdls {



/// \brief Elimination schedule of the tiled factorization.
enum class TiledLUppSchedule {
    RightLooking, ///< Factor the diagonal tile, push updates into the trailing matrix.
    LeftLooking   ///< Pull updates from prior tiles when a tile is visited.
};



/// \brief Default compile-time knobs of the TiledLUpp solvers.
/// \tparam T scalar type (float or double)
template<typename T>
struct TiledLUppDefaultConfig {

    /// Tile extent: the matrix is processed as a grid of tile_size x
    /// tile_size register tiles. This is the main performance axis of the
    /// solvers - tune it per system dimension (measured optima in the
    /// source project: 3, 4 or 6 depending on N). The static TiledLUpp solver
    /// requires 2 <= tile_size <= N; the dynamic TiledLUpp solver only requires
    /// tile_size >= 2 (it may exceed n).
    static constexpr int tile_size = 3;

    /// Elimination schedule of the tiled factorization, see
    /// TiledLUppSchedule.
    static constexpr TiledLUppSchedule schedule = TiledLUppSchedule::RightLooking;

    /// Acceptable-pivot threshold of the out-of-tile search. An in-tile
    /// pivot candidate whose magnitude reaches this value is accepted
    /// without looking outside the tile; below it, the search extends to
    /// the rows under the tile (out-of-tile pivoting) and the best
    /// corrected candidate wins.
    static constexpr T oot_threshold = std::is_same_v<T, float> ? T(1e-4f) : T(1e-10);

    /// Singularity floor: the factorization is declared singular when even
    /// the best candidate of the out-of-tile recovery stays below it. The
    /// floor only guards the recovery path (a pivot reaching oot_threshold
    /// is accepted directly), so it must not exceed oot_threshold; the
    /// solvers enforce this contract at compile time. `numeric_limits<T>::min()`
    /// rejects only a zero/subnormal pivot - a genuine structural
    /// singularity. A merely small pivot is kept on purpose: the loss of
    /// stability is surfaced by the backward error and overflow is caught
    /// downstream by the caller, whereas an absolute floor wrongly flags
    /// well-conditioned matrices at small scale.
    static constexpr T singular_eps = std::numeric_limits<T>::min();

    /// Out-of-tile pivot search strategy. When true, the below-tile scan
    /// stops at the first candidate whose corrected magnitude reaches
    /// oot_threshold instead of scanning the whole panel for the maximum;
    /// the running maximum is still kept as the fallback when no candidate
    /// is acceptable. Cheaper in the OOT-heavy regime - especially
    /// left-looking, where every candidate replays the prior tiles - at
    /// the cost of a possibly smaller (but still >= oot_threshold) pivot.
    /// Set false to restore the full-panel partial-pivoting scan.
    static constexpr bool oot_first_acceptable = true;

    /// Unroll policy of the in-tile scalar loops, applied through a
    /// two-branch `if constexpr` (the pragma dialect itself lives in
    /// core/macros.hpp). true: loops indexing register tiles carry a
    /// forced-unroll pragma - the guard that keeps tiles in registers on
    /// GPU backends, where a rolled loop indexes the tile dynamically and
    /// demotes it to slow local memory. false: no unroll pragma anywhere -
    /// faster compiles, GPU performance not guaranteed. Outer tile-sweep
    /// loops never carry a pragma in either branch.
    static constexpr bool unroll_inner = true;
};



/// \brief Convenience configuration selecting the tile size and the
/// schedule while keeping every other knob at its default.
/// \tparam T     scalar type (float or double)
/// \tparam TS    tile size (int)
/// \tparam Schedule elimination schedule (RightLooking or LeftLooking)
template<typename T, int TS, TiledLUppSchedule Schedule = TiledLUppSchedule::RightLooking>
struct TiledLUppConfig : TiledLUppDefaultConfig<T> {
    static constexpr int tile_size = TS; ///< tile size (int)
    static constexpr TiledLUppSchedule schedule =
        Schedule; ///< elimination schedule (RightLooking or LeftLooking)
};



} // namespace tdls



#endif // TDLS_SOLVERS_TILED_LUPP_CONFIG_HPP
