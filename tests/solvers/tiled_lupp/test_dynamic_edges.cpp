/// \file
/// \brief Edge suite of the dynamic TiledLUpp solver.
/// \author Tristan Chenaille
///
/// Covers what only the dynamic TiledLUpp solver can reach: the tile-grid helper
/// functions on assorted runtime shapes, the minimal dimension, and a
/// dimension well beyond the static test grid, anchored on the backward
/// error against the reference LU.

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"
#include "reference_lu.hpp"

namespace {

/// \brief Anchors the dynamic TiledLUpp solver on the backward error for one
/// runtime shape.
/// \tparam TS tile size
/// \param[in] n         system dimension
/// \param[in] count     number of systems
/// \param[in] tolerance backward-error bound
/// \param[in] seed      generator seed
template<int TS>
void anchor(const int n, const int count, const double tolerance, const std::uint64_t seed) {
    using Solver     = tdls::TiledLUppSolverDynamic<double, tdls::TiledLUppConfig<double, TS>>;
    const auto batch = tdls_tests::make_batch<double>(n, count, seed, 0.5);
    std::vector<double> A(static_cast<std::size_t>(n) * n), x(n);
    std::vector<int> piv(n);
    double be_max = 0.0;
    for (int s = 0; s < count; ++s) {
        std::copy(batch.matrix(s), batch.matrix(s) + n * n, A.begin());
        TDLS_CHECK(Solver::solve(n, A.data(), 1, piv.data(), 1, batch.rhs(s), x.data(), 1));
        be_max = std::max(be_max,
                          tdls_tests::backward_error(batch.matrix(s), x.data(), batch.rhs(s), n));
    }
    TDLS_CHECK_LE(be_max, tolerance);
}

} // namespace

TDLS_TEST_CASE("tiledlupp/dynamic-edges/tile-grid-helpers") {
    using S4 = tdls::TiledLUppSolverDynamic<double, tdls::TiledLUppConfig<double, 4>>;
    // Divisible, partial-tile and single-tile grids.
    TDLS_CHECK(S4::num_tiles(4) == 1);
    TDLS_CHECK(S4::num_tiles(8) == 2);
    TDLS_CHECK(S4::num_tiles(9) == 3);
    TDLS_CHECK(S4::num_tiles(2) == 1);
    TDLS_CHECK(S4::tile_extent(0, 9) == 4);
    TDLS_CHECK(S4::tile_extent(4, 9) == 4);
    TDLS_CHECK(S4::tile_extent(8, 9) == 1);
    TDLS_CHECK(S4::tile_extent(0, 2) == 2);
    using S6 = tdls::TiledLUppSolverDynamic<double, tdls::TiledLUppConfig<double, 6>>;
    TDLS_CHECK(S6::num_tiles(13) == 3);
    TDLS_CHECK(S6::tile_extent(12, 13) == 1);
}

TDLS_TEST_CASE("tiledlupp/dynamic-edges/minimal-dimension/n=2,TS=2") {
    anchor<2>(2, 300, 1e-9, 230100);
}

TDLS_TEST_CASE("tiledlupp/dynamic-edges/single-partial-tile/n=3,TS=8") {
    anchor<8>(3, 300, 1e-9, 230200);
}

TDLS_TEST_CASE("tiledlupp/dynamic-edges/large-dimension/n=200,TS=6") {
    anchor<6>(200, 8, 1e-8, 230300);
}

TDLS_TEST_MAIN
