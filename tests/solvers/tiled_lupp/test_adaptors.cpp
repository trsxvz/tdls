/// \file
/// \brief Suite of the generic dense adaptors.
/// \author Tristan Chenaille
///
/// The adaptors are structural: they accept any type exposing the dense
/// contract, without naming any external library. The suite exercises the
/// three recognized data() shapes through minimal mock types (contiguous
/// object, strided view returning a (pointer, stride) pair, strided view
/// with a separate getStride()), checks the residency classification at
/// compile time, rejects a gather-like type, and requires every adaptor
/// call to reproduce the raw pointer API bitwise.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

constexpr int N = 12;

/// \brief Indexing policy of the matrix mocks, row-major contiguous.
struct MockMatrixPolicy {
    using size_type            = int;
    static constexpr int arity = 2;
    constexpr int size(const int d) const {
        return d >= 0 ? N : N;
    }
    constexpr int getIndex(const int i, const int j) const {
        return i * N + j;
    }
};

/// \brief Indexing policy of the vector mocks, contiguous.
struct MockVectorPolicy {
    using size_type            = int;
    static constexpr int arity = 1;
    constexpr int size(const int) const {
        return N;
    }
    constexpr int getIndex(const int i) const {
        return i;
    }
};

/// \brief Contiguous matrix mock: data() returns a plain pointer.
struct MockMatrix {
    using indexing_policy = MockMatrixPolicy;
    double v[N * N];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Contiguous vector mock.
struct MockVector {
    using indexing_policy = MockVectorPolicy;
    double v[N];
    double* data() {
        return v;
    }
    const double* data() const {
        return v;
    }
};

/// \brief Contiguous int vector mock, used as a dense pivot.
struct MockIntVector {
    using indexing_policy = MockVectorPolicy;
    int v[N];
    int* data() {
        return v;
    }
    const int* data() const {
        return v;
    }
};

/// \brief Strided matrix view mock: data() returns a (pointer, stride)
/// pair, the shape adopted by tfel::math::StridedCoalescedView.
struct MockPairMatrixView {
    using indexing_policy = MockMatrixPolicy;
    double* pointer;
    std::size_t stride_value;
    std::pair<double*, std::size_t> data() const {
        return {pointer, stride_value};
    }
};

/// \brief Strided vector view mock: plain data() plus a separate
/// getStride().
struct MockGetStrideVectorView {
    using indexing_policy = MockVectorPolicy;
    double* pointer;
    int stride_value;
    double* data() const {
        return pointer;
    }
    int getStride() const {
        return stride_value;
    }
};

/// \brief Gather-like mock: one pointer per element and no data() at
/// all, hence outside the dense contract.
struct MockGatherView {
    using indexing_policy = MockMatrixPolicy;
    double* pointers[N * N];
};

// Compile-time classification contract.
static_assert(!tdls::detail::is_dense_v<MockGatherView>);
static_assert(tdls::storage_traits<MockMatrix>::is_internal);
static_assert(tdls::storage_traits<MockVector>::is_internal);
static_assert(!tdls::storage_traits<MockPairMatrixView>::is_internal);
static_assert(!tdls::storage_traits<MockGetStrideVectorView>::is_internal);

using RawSolver = tdls::TiledLUppSolverStatic<double, N, tdls::TiledLUppConfig<double, 3>>;

} // namespace

TDLS_TEST_CASE("tiledlupp/adaptors/contiguous-mocks-reproduce-raw-internal") {
    tdls_tests::UniformGenerator gen(210100, 0.5);
    for (int repeat = 0; repeat < 100; ++repeat) {
        MockMatrix A;
        MockVector b, x;
        double A_raw[N * N], b_raw[N], x_raw[N];
        int piv[N], piv_raw[N];
        for (int e = 0; e < N * N; ++e)
            A.v[e] = A_raw[e] = gen.next();
        for (int i = 0; i < N; ++i)
            b.v[i] = b_raw[i] = gen.next();
        const bool ok = tdls::solve(A, piv, b, x);
        const bool ok_raw =
            RawSolver::solve<true, true, true>(A_raw, 1, piv_raw, 1, b_raw, x_raw, 1);
        TDLS_CHECK(ok == ok_raw);
        TDLS_CHECK_BITWISE(A.v, A_raw, static_cast<std::size_t>(N) * N);
        TDLS_CHECK_BITWISE(piv, piv_raw, static_cast<std::size_t>(N));
        TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));
    }
}

TDLS_TEST_CASE("tiledlupp/adaptors/strided-mocks-reproduce-raw-external") {
    // Matrix through the pair shape, vectors through the getStride shape,
    // all mapped on one SoA batch.
    constexpr int count = 64;
    tdls_tests::UniformGenerator gen(210200, 0.5);
    std::vector<double> gA(static_cast<std::size_t>(N) * N * count);
    std::vector<double> gb(static_cast<std::size_t>(N) * count);
    std::vector<double> gx(static_cast<std::size_t>(N) * count, 0.0);
    for (auto& v : gA)
        v = gen.next();
    for (auto& v : gb)
        v = gen.next();
    std::vector<double> gA_raw(gA), gx_raw(static_cast<std::size_t>(N) * count, 0.0);

    for (int s = 0; s < count; ++s) {
        MockPairMatrixView A{gA.data() + s, static_cast<std::size_t>(count)};
        MockGetStrideVectorView b{gb.data() + s, count}, x{gx.data() + s, count};
        int piv[N], piv_raw[N];
        const bool ok = tdls::solve(A, piv, b, x);
        const bool ok_raw =
            RawSolver::factorize<true, false>(gA_raw.data() + s, count, piv_raw, 1) &&
            (RawSolver::substitute<false, true, false>(gA_raw.data() + s, count, piv_raw, 1,
                                                       gb.data() + s, gx_raw.data() + s, count),
             true);
        TDLS_CHECK(ok == ok_raw);
        TDLS_CHECK_BITWISE(piv, piv_raw, static_cast<std::size_t>(N));
    }
    TDLS_CHECK_BITWISE(gA.data(), gA_raw.data(), gA.size());
    TDLS_CHECK_BITWISE(gx.data(), gx_raw.data(), gx.size());
}

TDLS_TEST_CASE("tiledlupp/adaptors/dense-int-pivot") {
    tdls_tests::UniformGenerator gen(210300, 0.5);
    MockMatrix A;
    double A_raw[N * N];
    MockIntVector piv;
    int piv_raw[N];
    for (int e = 0; e < N * N; ++e)
        A.v[e] = A_raw[e] = gen.next();
    const bool ok     = tdls::factorize(A, piv);
    const bool ok_raw = RawSolver::factorize<true, true>(A_raw, 1, piv_raw, 1);
    TDLS_CHECK(ok == ok_raw);
    TDLS_CHECK_BITWISE(A.v, A_raw, static_cast<std::size_t>(N) * N);
    TDLS_CHECK_BITWISE(piv.v, piv_raw, static_cast<std::size_t>(N));
}

TDLS_TEST_CASE("tiledlupp/adaptors/substitution-entry-points-reproduce-raw") {
    tdls_tests::UniformGenerator gen(210400, 0.5);
    MockMatrix A;
    MockVector b, x;
    double A_raw[N * N], b_raw[N], x_raw[N];
    int piv[N], piv_raw[N];
    for (int e = 0; e < N * N; ++e)
        A.v[e] = A_raw[e] = gen.next();
    for (int i = 0; i < N; ++i)
        b.v[i] = b_raw[i] = gen.next();

    TDLS_CHECK(tdls::factorize(A, piv));
    TDLS_CHECK((RawSolver::factorize<true, true>(A_raw, 1, piv_raw, 1)));

    tdls::substitute(A, piv, b, x);
    RawSolver::substitute<true, true, true>(A_raw, 1, piv_raw, 1, b_raw, x_raw, 1);
    TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));

    MockVector y;
    double y_raw[N];
    for (int i = 0; i < N; ++i) {
        y.v[i]   = b.v[i];
        y_raw[i] = b_raw[i];
    }
    tdls::substitute_inplace(A, piv, y);
    RawSolver::substitute_inplace<true, true, true>(A_raw, 1, piv_raw, 1, y_raw, 1);
    TDLS_CHECK_BITWISE(y.v, y_raw, static_cast<std::size_t>(N));

    for (int c = 0; c < N; ++c) {
        tdls::substitute_canonical(A, piv, c, x);
        RawSolver::substitute_canonical<true, true, true>(A_raw, 1, piv_raw, 1, c, x_raw, 1);
        TDLS_CHECK_BITWISE(x.v, x_raw, static_cast<std::size_t>(N));
    }

    MockMatrix A2;
    double A2_raw[N * N];
    for (int e = 0; e < N * N; ++e)
        A2.v[e] = A2_raw[e] = A.v[e] * 0.5 + 0.25;
    MockVector z;
    double z_raw[N];
    for (int i = 0; i < N; ++i) {
        z.v[i]   = b.v[i];
        z_raw[i] = b_raw[i];
    }
    const bool ok_fused = tdls::solve_fused(A2, piv, z);
    const bool ok_fused_raw =
        RawSolver::solve_fused<true, true, true>(A2_raw, 1, piv_raw, 1, z_raw, 1);
    TDLS_CHECK(ok_fused == ok_fused_raw);
    TDLS_CHECK_BITWISE(z.v, z_raw, static_cast<std::size_t>(N));
}

TDLS_TEST_MAIN
