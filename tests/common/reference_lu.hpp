#ifndef TDLS_TESTS_COMMON_REFERENCE_LU_HPP
#define TDLS_TESTS_COMMON_REFERENCE_LU_HPP



/// \file
/// \brief Independent reference solver and error metric of the test
/// suites, shared later with the benchmark harness.
/// \author Tristan Chenaille
///
/// Deliberately naive dense LU with partial pivoting on contiguous
/// row-major storage, written independently from the TiledLUpp solvers:
/// scalar elimination, physical row swaps, true pivot values on the
/// diagonal (not their reciprocals). Results are therefore NOT bitwise
/// comparable with the TiledLUpp solvers; the anchor metric is the normwise
/// backward error below.
///
/// The singularity criterion mirrors the TiledLUpp solvers (best pivot below
/// numeric_limits::min(), i.e. zero or subnormal) so that verdicts can be
/// compared exactly.



#include <cmath>
#include <limits>



namespace tdls_tests {



/// \brief Factor a contiguous row-major matrix in place with partial
/// pivoting, LAPACK-style: unit lower L below the diagonal, U on and
/// above it, physical row swaps recorded in piv.
/// \tparam T scalar type
/// \param[in,out] A   matrix, n x n contiguous row-major
/// \param[out]    piv row swapped with row k at step k (piv[k] >= k)
/// \param[in]     n   system dimension
/// \return false on a singular matrix.
template<typename T>
inline bool reference_factorize(T* A, int* piv, const int n) {
    for (int k = 0; k < n; ++k) {
        int best_row = k;
        T best       = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            const T v = std::fabs(A[i * n + k]);
            if (v > best) {
                best     = v;
                best_row = i;
            }
        }
        if (best < std::numeric_limits<T>::min()) return false;
        piv[k] = best_row;
        if (best_row != k) {
            for (int j = 0; j < n; ++j) {
                const T tmp         = A[k * n + j];
                A[k * n + j]        = A[best_row * n + j];
                A[best_row * n + j] = tmp;
            }
        }
        for (int i = k + 1; i < n; ++i) {
            A[i * n + k] /= A[k * n + k];
            for (int j = k + 1; j < n; ++j)
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
        }
    }
    return true;
}

/// \brief Solve in place from a reference factorization: x holds the
/// right-hand side on entry and the solution on exit.
/// \tparam T scalar type
/// \param[in]     A   factored matrix from reference_factorize
/// \param[in]     piv row swaps recorded by reference_factorize
/// \param[in,out] x   right-hand side, then solution
/// \param[in]     n   system dimension
template<typename T>
inline void reference_substitute(const T* A, const int* piv, T* x, const int n) {
    for (int k = 0; k < n; ++k) {
        if (piv[k] != k) {
            const T tmp = x[k];
            x[k]        = x[piv[k]];
            x[piv[k]]   = tmp;
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i)
            x[i] -= A[i * n + k] * x[k];
    }
    for (int k = n - 1; k >= 0; --k) {
        for (int j = k + 1; j < n; ++j)
            x[k] -= A[k * n + j] * x[j];
        x[k] /= A[k * n + k];
    }
}

/// \brief Factor and solve in one call.
/// \tparam T scalar type
/// \param[in,out] A   matrix on entry, factored on exit
/// \param[out]    piv row swaps
/// \param[in]     b   right-hand side
/// \param[out]    x   solution
/// \param[in]     n   system dimension
/// \return false on a singular matrix.
template<typename T>
inline bool reference_solve(T* A, int* piv, const T* b, T* x, const int n) {
    for (int i = 0; i < n; ++i)
        x[i] = b[i];
    if (!reference_factorize(A, piv, n)) return false;
    reference_substitute(A, piv, x, n);
    return true;
}

/// \brief Normwise backward error of a computed solution,
/// |A0 x - b|_inf / (|A0|_max |x|_inf + |b|_inf), accumulated in double
/// regardless of T. The thresholds used by the suites are calibrated for
/// this exact formula.
/// \tparam T scalar type
/// \param[in] A0 original (unfactored) matrix, contiguous row-major
/// \param[in] x  computed solution
/// \param[in] b  right-hand side
/// \param[in] n  system dimension
/// \return the backward error
template<typename T>
inline double backward_error(const T* A0, const T* x, const T* b, const int n) {
    double residual_max = 0.0;
    double a_max        = 0.0;
    double x_max        = 0.0;
    double b_max        = 0.0;
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            acc += static_cast<double>(A0[i * n + j]) * static_cast<double>(x[j]);
            a_max = std::fmax(a_max, std::fabs(static_cast<double>(A0[i * n + j])));
        }
        residual_max = std::fmax(residual_max, std::fabs(acc - static_cast<double>(b[i])));
        x_max        = std::fmax(x_max, std::fabs(static_cast<double>(x[i])));
        b_max        = std::fmax(b_max, std::fabs(static_cast<double>(b[i])));
    }
    return residual_max / (a_max * x_max + b_max);
}



} // namespace tdls_tests



#endif // TDLS_TESTS_COMMON_REFERENCE_LU_HPP
