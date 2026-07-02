/// \file
/// \brief Suite of the tile-size axis.
/// \author Tristan Chenaille
///
/// Different tile sizes execute different arithmetic sequences, so tile
/// sizes cannot be compared bitwise to each other. Each tile size is
/// therefore anchored on the backward error against the reference LU,
/// and bridged bitwise against the dynamic TiledLUpp solver at equal shape. The
/// sweeps include divisible grids, trailing tiles and the TS = N corner;
/// the dynamic-only TS > n corner is anchored by the dynamic oracle
/// suite.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Anchors one (N, TS) cell on the backward error and bridges it
/// bitwise against the dynamic TiledLUpp solver.
/// \tparam T  scalar type
/// \tparam N  system dimension
/// \tparam TS tile size
/// \param[in] count     number of systems
/// \param[in] bound     half-width of the entry distribution
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<typename T, int N, int TS>
void tile_size_case(const int count, const double bound, const double tolerance,
                    const std::uint64_t seed) {
    using Config     = tdls::TiledLUppConfig<T, TS>;
    using Static     = tdls::TiledLUppSolverStatic<T, N, Config>;
    using Dynamic    = tdls::TiledLUppSolverDynamic<T, Config>;
    const auto batch = tdls_tests::make_batch<T>(N, count, seed, bound);

    double be_max = 0.0;
    std::vector<T> A_static(static_cast<std::size_t>(N) * N);
    std::vector<T> A_dynamic(static_cast<std::size_t>(N) * N);
    std::vector<T> x_static(N), x_dynamic(N);
    std::vector<int> piv_static(N), piv_dynamic(N);
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_static.begin());
        std::copy(batch.matrix(s), batch.matrix(s) + N * N, A_dynamic.begin());
        const bool ok_static = Static::template solve<false, false, false>(
            A_static.data(), 1, piv_static.data(), 1, batch.rhs(s), x_static.data(), 1);
        const bool ok_dynamic = Dynamic::solve(N, A_dynamic.data(), 1, piv_dynamic.data(), 1,
                                               batch.rhs(s), x_dynamic.data(), 1);
        TDLS_CHECK(ok_static == ok_dynamic);
        if (!ok_static || !ok_dynamic) continue;
        be_max = std::max(
            be_max, tdls_tests::backward_error(batch.matrix(s), x_static.data(), batch.rhs(s), N));
        TDLS_CHECK_BITWISE(A_static.data(), A_dynamic.data(), static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv_static.data(), piv_dynamic.data(), static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x_static.data(), x_dynamic.data(), static_cast<std::size_t>(N));
    }
    TDLS_CHECK_LE(be_max, tolerance);
}

} // namespace

/// Emits the tile-size case of one (N, TS) cell.
#define TDLS_TILE_SIZE_CASE(N, TS, COUNT, SEED)                                                    \
    TDLS_TEST_CASE("tiledlupp/tile-sizes/double/N=" #N ",TS=" #TS ",default") {                    \
        tile_size_case<double, N, TS>(COUNT, 0.5, 1e-9, SEED);                                     \
    }

// Divisible grid: every divisor tile size of N = 12, including TS = N.
TDLS_TILE_SIZE_CASE(12, 2, 200, 170122)
TDLS_TILE_SIZE_CASE(12, 3, 200, 170123)
TDLS_TILE_SIZE_CASE(12, 4, 200, 170124)
TDLS_TILE_SIZE_CASE(12, 6, 200, 170126)
TDLS_TILE_SIZE_CASE(12, 12, 200, 170132)
// Non-divisor tile sizes on N = 12: trailing tiles of every extent.
TDLS_TILE_SIZE_CASE(12, 5, 200, 170125)
TDLS_TILE_SIZE_CASE(12, 7, 200, 170127)
// Prime dimension: every tile size leaves a trailing tile.
TDLS_TILE_SIZE_CASE(13, 2, 200, 170232)
TDLS_TILE_SIZE_CASE(13, 4, 200, 170234)
TDLS_TILE_SIZE_CASE(13, 5, 200, 170235)
TDLS_TILE_SIZE_CASE(13, 6, 200, 170236)
TDLS_TILE_SIZE_CASE(13, 13, 200, 170243)

TDLS_TEST_MAIN
