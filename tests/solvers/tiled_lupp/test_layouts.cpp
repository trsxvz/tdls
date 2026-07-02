/// \file
/// \brief Bridge suite: batched layouts are equivalent.
/// \author Tristan Chenaille
///
/// In external mode, the TiledLUpp solvers see every batched layout through the
/// same (pre-offset pointer, element stride) pair: AoS is stride 1, SoA
/// is stride count, AoSoA is stride W. On identical inputs, the three
/// layouts must therefore produce bitwise-identical factored matrices,
/// pivots and solutions, for the matrix, the right-hand side and the
/// pivot simultaneously. SoA is the baseline; AoS and AoSoA (W = 4) are
/// compared to it, with both solvers.

#include <cstdint>
#include <vector>

#include <tdls/tdls.hpp>

#include "generators.hpp"
#include "harness.hpp"

namespace {

/// \brief Identifies one batched layout.
enum class Layout { soa, aos, aosoa };

/// \brief Buffer index of (element, system) under the given layout.
/// \param[in] layout      the layout
/// \param[in] element     flat element index inside the object
/// \param[in] system      system index
/// \param[in] object_size elements per object
/// \param[in] count       systems in the buffer
/// \return the buffer index
std::size_t layout_index(const Layout layout, const std::size_t element, const std::size_t system,
                         const std::size_t object_size, const std::size_t count) {
    switch (layout) {
    case Layout::soa:
        return tdls_tests::soa_index(element, system, count);
    case Layout::aos:
        return tdls_tests::aos_index(element, system, object_size);
    default:
        return tdls_tests::aosoa_index(element, system, object_size, 4);
    }
}

/// \brief Element stride of the given layout.
/// \param[in] layout the layout
/// \param[in] count  systems in the buffer
/// \return the stride
int layout_stride(const Layout layout, const int count) {
    switch (layout) {
    case Layout::soa:
        return count;
    case Layout::aos:
        return 1;
    default:
        return 4;
    }
}

/// \brief Factors and solves in place every system of a batch scattered
/// under the given layout (matrix, right-hand side and pivot all
/// external), then gathers the outputs contiguous.
/// \tparam Solver solver type providing the static entry points
/// \tparam T      scalar type
/// \tparam N      system dimension
/// \param[in]  batch  input systems (count must be a multiple of 4)
/// \param[in]  layout the layout under test
/// \param[out] A_out  factored matrices, contiguous per system
/// \param[out] piv_out pivots, contiguous per system
/// \param[out] x_out  solutions, contiguous per system
/// \param[out] verdicts one flag per system
template<typename Solver, typename T, int N>
void run_layout(const tdls_tests::SystemBatch<T>& batch, const Layout layout, std::vector<T>& A_out,
                std::vector<int>& piv_out, std::vector<T>& x_out, std::vector<char>& verdicts) {
    const int count = batch.count;
    std::vector<T> A_buf(static_cast<std::size_t>(N) * N * count);
    std::vector<T> x_buf(static_cast<std::size_t>(N) * count);
    std::vector<int> piv_buf(static_cast<std::size_t>(N) * count);
    for (int s = 0; s < count; ++s) {
        for (int e = 0; e < N * N; ++e)
            A_buf[layout_index(layout, e, s, N * N, count)] = batch.matrix(s)[e];
        for (int i = 0; i < N; ++i)
            x_buf[layout_index(layout, i, s, N, count)] = batch.rhs(s)[i];
    }
    const int a_stride = layout_stride(layout, count);
    for (int s = 0; s < count; ++s) {
        T* A          = A_buf.data() + layout_index(layout, 0, s, N * N, count);
        T* x          = x_buf.data() + layout_index(layout, 0, s, N, count);
        int* piv      = piv_buf.data() + layout_index(layout, 0, s, N, count);
        const bool ok = Solver::template factorize<false, false>(A, a_stride, piv, a_stride);
        if (ok)
            Solver::template substitute_inplace<false, false, false>(A, a_stride, piv, a_stride, x,
                                                                     a_stride);
        verdicts[s] = ok ? 1 : 0;
    }
    for (int s = 0; s < count; ++s) {
        for (int e = 0; e < N * N; ++e)
            A_out[static_cast<std::size_t>(s) * N * N + e] =
                A_buf[layout_index(layout, e, s, N * N, count)];
        for (int i = 0; i < N; ++i) {
            piv_out[static_cast<std::size_t>(s) * N + i] =
                piv_buf[layout_index(layout, i, s, N, count)];
            x_out[static_cast<std::size_t>(s) * N + i] =
                x_buf[layout_index(layout, i, s, N, count)];
        }
    }
}

/// \brief Compares the AoS and AoSoA layouts to the SoA baseline over a
/// reproducible batch.
/// \tparam Solver solver type providing the static entry points
/// \tparam T      scalar type
/// \tparam N      system dimension
/// \param[in] count number of systems (multiple of 4)
/// \param[in] bound half-width of the entry distribution
/// \param[in] seed  generator seed
template<typename Solver, typename T, int N>
void layouts_case(const int count, const double bound, const std::uint64_t seed) {
    const auto batch          = tdls_tests::make_batch<T>(N, count, seed, bound);
    const std::size_t a_total = static_cast<std::size_t>(count) * N * N;
    const std::size_t v_total = static_cast<std::size_t>(count) * N;

    std::vector<T> A_ref(a_total), x_ref(v_total), A_tst(a_total), x_tst(v_total);
    std::vector<int> piv_ref(v_total), piv_tst(v_total);
    std::vector<char> ok_ref(count), ok_tst(count);
    run_layout<Solver, T, N>(batch, Layout::soa, A_ref, piv_ref, x_ref, ok_ref);
    for (const Layout layout : {Layout::aos, Layout::aosoa}) {
        run_layout<Solver, T, N>(batch, layout, A_tst, piv_tst, x_tst, ok_tst);
        TDLS_CHECK_BITWISE(ok_ref.data(), ok_tst.data(), static_cast<std::size_t>(count));
        TDLS_CHECK_BITWISE(A_ref.data(), A_tst.data(), a_total);
        TDLS_CHECK_BITWISE(piv_ref.data(), piv_tst.data(), v_total);
        TDLS_CHECK_BITWISE(x_ref.data(), x_tst.data(), v_total);
    }
}

/// \brief Static-solver front end of layouts_case.
template<typename T, int N, int TS, tdls::TiledLUppSchedule Sched>
void layouts_case_static(const int count, const double bound, const std::uint64_t seed) {
    using Solver = tdls::TiledLUppSolverStatic<T, N, tdls::TiledLUppConfig<T, TS, Sched>>;
    layouts_case<Solver, T, N>(count, bound, seed);
}

/// \brief Adapts the dynamic TiledLUpp solver to the static entry-point shape used
/// by run_layout, so both solvers share the same layout plumbing.
template<typename T, int N, int TS>
struct DynamicFrontEnd {
    using Solver = tdls::TiledLUppSolverDynamic<T, tdls::TiledLUppConfig<T, TS>>;
    template<bool, bool>
    static bool factorize(T* A, const int a_stride, int* piv, const int piv_stride) {
        return Solver::factorize(N, A, a_stride, piv, piv_stride);
    }
    template<bool, bool, bool>
    static void substitute_inplace(const T* A, const int a_stride, const int* piv,
                                   const int piv_stride, T* x, const int rhs_stride) {
        Solver::substitute_inplace(N, A, a_stride, piv, piv_stride, x, rhs_stride);
    }
};

} // namespace

TDLS_TEST_CASE("tiledlupp/bridge/layouts/static/double/N=12,TS=3,RL,default") {
    layouts_case_static<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 140100);
}
TDLS_TEST_CASE("tiledlupp/bridge/layouts/static/double/N=12,TS=3,RL,stress") {
    layouts_case_static<double, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 5e-10, 140200);
}
TDLS_TEST_CASE("tiledlupp/bridge/layouts/static/double/N=13,TS=6,LL,default") {
    layouts_case_static<double, 13, 6, tdls::TiledLUppSchedule::LeftLooking>(200, 0.5, 140300);
}
TDLS_TEST_CASE("tiledlupp/bridge/layouts/static/double/N=33,TS=5,RL,default") {
    layouts_case_static<double, 33, 5, tdls::TiledLUppSchedule::RightLooking>(100, 0.5, 140400);
}
TDLS_TEST_CASE("tiledlupp/bridge/layouts/static/float/N=12,TS=3,RL,default") {
    layouts_case_static<float, 12, 3, tdls::TiledLUppSchedule::RightLooking>(200, 0.5, 140500);
}
TDLS_TEST_CASE("tiledlupp/bridge/layouts/dynamic/double/n=13,TS=6,default") {
    layouts_case<DynamicFrontEnd<double, 13, 6>, double, 13>(200, 0.5, 140600);
}

TDLS_TEST_MAIN
