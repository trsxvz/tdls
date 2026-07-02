#ifndef TDLS_ADAPTORS_HPP
#define TDLS_ADAPTORS_HPP



/// \file
/// \brief Generic adaptors: call the TiledLUpp solvers directly on dense
/// math objects (matrices, vectors, views) instead of raw pointers.
/// \author Tristan Chenaille
///
/// The adaptors accept any object matching a small STRUCTURAL contract -
/// no external library is named or included. Two families are recognized:
///
///   - contiguous fixed-size objects and views: expose data() and an
///     indexing_policy type whose extents are constexpr (this matches
///     tfel::math tmatrix / tvector / View, among others);
///   - element-strided views: expose their runtime stride either
///     through a data() member returning a (pointer, stride) pair (this
///     matches tfel::math StridedCoalescedView, i.e. the map_strided SoA
///     views) or through a stride() / getStride() member next to a plain
///     data().
///
/// The system dimension is deduced at compile time from the indexing
/// policy, and the residency template booleans of the raw API are
/// inferred from the argument types (contiguous object -> internal,
/// strided view -> external). Each argument is unwrapped independently,
/// so residencies can be mixed freely.
///
/// Two families are rejected at compile time with an explicit message:
/// row-strided matrix views (sub-matrix views, whose stride between rows
/// is unrelated to the column count) cannot be expressed by the TiledLUpp solvers'
/// single-stride addressing, and gather views holding one pointer per
/// element expose no data() at all.
///
/// The detection can be overridden for exotic types by specializing
/// tdls::storage_traits.
///
/// The entry points currently dispatch to the TiledLUpp solver family;
/// a family selection parameter can join when another family lands.



#include <cstddef>
#include <type_traits>

#include <tdls/core/macros.hpp>
#include <tdls/solvers/tiled_lupp/solver_static.hpp>



namespace tdls {



namespace detail {

/// \brief Detects a data() member returning a pointer.
template<typename T, typename = void>
struct has_data : std::false_type {};
template<typename T>
struct has_data<T, std::void_t<decltype(std::declval<const T&>().data())>>
    : std::is_pointer<decltype(std::declval<const T&>().data())> {};

/// \brief Detects a data() member returning a (pointer, stride) pair -
/// the shape of strided views, whose data pointer must not be mistaken
/// for contiguous storage.
template<typename T, typename = void>
struct has_pair_data : std::false_type {};
template<typename T>
struct has_pair_data<T, std::void_t<decltype(std::declval<const T&>().data().first),
                                    decltype(std::declval<const T&>().data().second)>>
    : std::bool_constant<std::is_pointer_v<decltype(std::declval<const T&>().data().first)> &&
                         std::is_integral_v<decltype(std::declval<const T&>().data().second)>> {};

/// \brief Element pointer type of a dense object, const flavour.
template<typename T, bool = has_pair_data<T>::value>
struct const_data_pointer {
    using type = decltype(std::declval<const T&>().data());
};
template<typename T>
struct const_data_pointer<T, true> {
    using type = decltype(std::declval<const T&>().data().first);
};

/// \brief Element pointer type of a dense object, mutable flavour.
template<typename T, bool = has_pair_data<T>::value>
struct mutable_data_pointer {
    using type = decltype(std::declval<T&>().data());
};
template<typename T>
struct mutable_data_pointer<T, true> {
    using type = decltype(std::declval<T&>().data().first);
};

/// \brief Detects a stride() member.
template<typename T, typename = void>
struct has_stride : std::false_type {};
template<typename T>
struct has_stride<T, std::void_t<decltype(std::declval<const T&>().stride())>> : std::true_type {};

/// \brief Detects a getStride() member.
template<typename T, typename = void>
struct has_get_stride : std::false_type {};
template<typename T>
struct has_get_stride<T, std::void_t<decltype(std::declval<const T&>().getStride())>>
    : std::true_type {};

/// \brief Detects a nested indexing_policy type.
template<typename T, typename = void>
struct has_indexing_policy : std::false_type {};
template<typename T>
struct has_indexing_policy<T, std::void_t<typename T::indexing_policy>> : std::true_type {};

/// \brief A type is "dense" when it exposes both a data pointer and an
/// indexing policy - the structural contract of the adaptors.
template<typename T>
inline constexpr bool is_dense_v =
    (has_data<T>::value || has_pair_data<T>::value) && has_indexing_policy<T>::value;

} // namespace detail



/// \brief Storage description of a dense object: element type, extents,
/// element pointer and element stride.
///
/// The primary template covers, structurally, every type exposing data()
/// and an indexing_policy (see the file documentation). Specialize it for
/// types that do not fit the structural contract.
/// \tparam DenseType dense object type (without cv-qualifiers/references)
template<typename DenseType, typename = void>
struct storage_traits;

template<typename DenseType>
struct storage_traits<DenseType, std::enable_if_t<detail::is_dense_v<DenseType>>> {
    //! \brief indexing policy of the object
    using indexing_policy = typename DenseType::indexing_policy;
    //! \brief a shorthand for the indexing size type
    using policy_size_type = typename indexing_policy::size_type;
    //! \brief element type (without cv-qualifiers)
    using value_type = std::remove_cv_t<
        std::remove_pointer_t<typename detail::const_data_pointer<DenseType>::type>>;
    //! \brief true when data() yields a mutable pointer
    static constexpr bool is_mutable = !std::is_const_v<
        std::remove_pointer_t<typename detail::mutable_data_pointer<DenseType>::type>>;
    //! \brief number of indices (1 = vector-like, 2 = matrix-like)
    static constexpr int arity = static_cast<int>(indexing_policy::arity);
    static_assert(arity == 1 || arity == 2, "tdls adaptors: only vector-like (arity 1) and "
                                            "matrix-like (arity 2) objects are supported");
    //! \brief extent along the first dimension
    static constexpr int extent0 = static_cast<int>(indexing_policy{}.size(0));
    //! \brief extent along the second dimension (1 for vectors)
    static constexpr int extent1 = (arity == 2) ? static_cast<int>(indexing_policy{}.size(1)) : 1;

    //! \return the compile-time part of the element stride, read from the
    //! indexing policy (distance between two column-consecutive elements)
    static constexpr int compute_policy_stride() {
        if constexpr (arity == 1) {
            return static_cast<int>(indexing_policy{}.getIndex(policy_size_type(1)));
        } else {
            return static_cast<int>(
                indexing_policy{}.getIndex(policy_size_type(0), policy_size_type(1)));
        }
    }
    //! \brief compile-time part of the element stride
    static constexpr int policy_stride = compute_policy_stride();

    //! \return true when the distance between rows is exactly extent1
    //! times the distance between columns (vectors: always true)
    static constexpr bool has_uniform_rows() {
        if constexpr (arity == 1) {
            return true;
        } else {
            return static_cast<int>(indexing_policy{}.getIndex(
                       policy_size_type(1), policy_size_type(0))) == extent1 * policy_stride;
        }
    }
    // Single-stride addressing requires uniform rows. Row-strided matrix
    // views (sub-matrix views) violate this and cannot be passed to the
    // solvers; solve the full matrix or use an element-strided
    // (coalesced) view instead.
    static_assert(has_uniform_rows(),
                  "tdls adaptors: row-strided matrix views (sub-matrix views) cannot "
                  "be expressed by the single-stride addressing of the TiledLUpp solvers");

    //! \brief true when the object carries a runtime element stride
    static constexpr bool has_runtime_stride = detail::has_pair_data<DenseType>::value ||
                                               detail::has_stride<DenseType>::value ||
                                               detail::has_get_stride<DenseType>::value;
    //! \brief true when the storage is compile-time contiguous: the raw
    //! API may then use the internal residency mode (direct indexing)
    static constexpr bool is_internal = (policy_stride == 1) && !has_runtime_stride;

    //! \return the pointer to the first element (const-ness is erased;
    //! mutating entry points check is_mutable beforehand)
    //! \param[in] o dense object
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static value_type* pointer(const DenseType& o) noexcept {
        if constexpr (detail::has_pair_data<DenseType>::value) {
            return const_cast<value_type*>(static_cast<const value_type*>(o.data().first));
        } else {
            return const_cast<value_type*>(static_cast<const value_type*>(o.data()));
        }
    }

    //! \return the total element stride (compile-time part times the
    //! runtime stride of the view, when any)
    //! \param[in] o dense object
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static int stride(const DenseType& o) noexcept {
        if constexpr (detail::has_pair_data<DenseType>::value) {
            return policy_stride * static_cast<int>(o.data().second);
        } else if constexpr (detail::has_stride<DenseType>::value) {
            return policy_stride * static_cast<int>(o.stride());
        } else if constexpr (detail::has_get_stride<DenseType>::value) {
            return policy_stride * static_cast<int>(o.getStride());
        } else {
            return policy_stride;
        }
    }
};



namespace detail {

/// \brief Common compile-time context of the adaptor entry points:
/// resolves the scalar type, the dimension and the TiledLUpp solver from the
/// matrix argument.
/// \tparam UserConfig user configuration (void = TiledLUppDefaultConfig)
/// \tparam MatrixType dense matrix type
template<typename UserConfig, typename MatrixType>
struct adaptor_context {
    //! \brief storage description of the matrix
    using mtraits = storage_traits<std::remove_cv_t<MatrixType>>;
    //! \brief scalar type of the system
    using scalar = typename mtraits::value_type;
    static_assert(std::is_floating_point_v<scalar>,
                  "tdls adaptors: the element type must be float or double");
    static_assert(mtraits::arity == 2, "tdls adaptors: A must be matrix-like");
    static_assert(mtraits::extent0 == mtraits::extent1, "tdls adaptors: A must be square");
    //! \brief system dimension
    static constexpr int N = mtraits::extent0;
    //! \brief resolved configuration
    using config =
        std::conditional_t<std::is_void_v<UserConfig>, TiledLUppDefaultConfig<scalar>, UserConfig>;
    //! \brief resolved solver
    using solver = TiledLUppSolverStatic<scalar, N, config>;
};

/// \brief Checks that a vector-like argument matches the system: arity 1,
/// extent N, same scalar type.
/// \tparam VectorType dense vector type
/// \tparam Scalar     scalar type of the system
/// \tparam N          system dimension
template<typename VectorType, typename Scalar, int N>
constexpr void check_vector() {
    using vtraits = storage_traits<std::remove_cv_t<VectorType>>;
    static_assert(vtraits::arity == 1, "tdls adaptors: expected a vector-like object");
    static_assert(vtraits::extent0 == N,
                  "tdls adaptors: vector extent does not match the system dimension");
    static_assert(std::is_same_v<typename vtraits::value_type, Scalar>,
                  "tdls adaptors: mixed scalar types");
}

/// \brief Checks that the right-hand side and the solution share one
/// residency and one compile-time stride: the raw API carries a single
/// rhs_stride for both. With runtime-strided views, both must be mapped
/// on the same batch (equal runtime strides), which is the caller's
/// responsibility.
/// \tparam RhsType      right-hand-side type
/// \tparam SolutionType solution type
template<typename RhsType, typename SolutionType>
constexpr void check_rhs_pair() {
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(bt::is_internal == xt::is_internal && bt::policy_stride == xt::policy_stride &&
                      bt::has_runtime_stride == xt::has_runtime_stride,
                  "tdls adaptors: b and x must share the same residency and "
                  "stride (the raw API carries a single rhs_stride for both)");
}

/// \brief Pivot argument unwrapping: accepts a raw int pointer/array
/// (treated as contiguous caller-local storage) or any dense int object.
/// \tparam PivotType pivot argument type (without references)
template<typename PivotType>
struct pivot_access {
    //! \brief true for a raw int pointer or int array
    static constexpr bool is_raw =
        std::is_pointer_v<std::decay_t<PivotType>> &&
        std::is_same_v<std::remove_cv_t<std::remove_pointer_t<std::decay_t<PivotType>>>, int>;
    //! \brief true for a dense int object
    static constexpr bool is_dense = detail::is_dense_v<std::remove_cv_t<PivotType>>;
    static_assert(is_raw || is_dense, "tdls adaptors: the pivot must be an int pointer/array or "
                                      "a dense int object");

    //! \brief true when the pivot storage is compile-time contiguous
    static constexpr bool is_internal = [] {
        if constexpr (is_raw) {
            return true;
        } else {
            return storage_traits<std::remove_cv_t<PivotType>>::is_internal;
        }
    }();

    //! \return the pivot element pointer
    //! \param[in] p pivot argument
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static int* pointer(const PivotType& p) noexcept {
        if constexpr (is_raw) {
            return const_cast<int*>(static_cast<const int*>(p));
        } else {
            using pt = storage_traits<std::remove_cv_t<PivotType>>;
            static_assert(std::is_same_v<typename pt::value_type, int>,
                          "tdls adaptors: the pivot element type must be int");
            return pt::pointer(p);
        }
    }

    //! \return the pivot element stride
    //! \param[in] p pivot argument
    TDLS_HOST_DEVICE TDLS_FORCEINLINE static int stride(const PivotType& p) noexcept {
        if constexpr (is_raw) {
            return 1;
        } else {
            return storage_traits<std::remove_cv_t<PivotType>>::stride(p);
        }
    }
};

} // namespace detail



/// \brief Factor a dense matrix object in place, A := P*L*U.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \return false on a singular matrix.
template<typename UserConfig = void, typename MatrixType, typename PivotType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE bool factorize(MatrixType& A, PivotType& piv) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    static_assert(mt::is_mutable, "tdls adaptors: factorize writes into A");
    return ctx::solver::template factorize<pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv));
}

/// \brief Solve A x = b on dense objects: factorize + substitute.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \param[in]     b   vector-like right-hand side, in original order
/// \param[out]    x   vector-like solution
/// \return false on a singular matrix.
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename RhsType,
         typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE bool solve(MatrixType& A, PivotType& piv, const RhsType& b,
                                             SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<RhsType, typename ctx::scalar, ctx::N>();
    detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
    detail::check_rhs_pair<RhsType, SolutionType>();
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(mt::is_mutable, "tdls adaptors: solve factors A in place");
    static_assert(xt::is_mutable, "tdls adaptors: solve writes into x");
    return ctx::solver::template solve<xt::is_internal, pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), bt::pointer(b),
        xt::pointer(x), xt::stride(x));
}

/// \brief Solve A y = y on dense objects with the fused factorization
/// (forward substitution folded into factorize, backward pass after).
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in,out] A   matrix-like object (factored in place)
/// \param[in,out] piv pivot storage: int pointer/array or dense int object
/// \param[in,out] y   vector-like right-hand side on entry, solution on exit
/// \return false on a singular matrix (y left partially updated).
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename VectorType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE bool solve_fused(MatrixType& A, PivotType& piv, VectorType& y) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<VectorType, typename ctx::scalar, ctx::N>();
    using yt = storage_traits<std::remove_cv_t<VectorType>>;
    static_assert(mt::is_mutable, "tdls adaptors: solve_fused factors A in place");
    static_assert(yt::is_mutable, "tdls adaptors: solve_fused writes into y");
    return ctx::solver::template solve_fused<yt::is_internal, pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), yt::pointer(y),
        yt::stride(y));
}

/// \brief Solve x := U^-1 L^-1 P b on dense objects, from a prior
/// factorize. b and x must not alias.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in]  A   factored matrix-like object
/// \param[in]  piv pivot storage produced by factorize
/// \param[in]  b   vector-like right-hand side, in original order
/// \param[out] x   vector-like solution
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename RhsType,
         typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE void substitute(const MatrixType& A, const PivotType& piv,
                                                  const RhsType& b, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<RhsType, typename ctx::scalar, ctx::N>();
    detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
    detail::check_rhs_pair<RhsType, SolutionType>();
    using bt = storage_traits<std::remove_cv_t<RhsType>>;
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute writes into x");
    ctx::solver::template substitute<xt::is_internal, pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), bt::pointer(b),
        xt::pointer(x), xt::stride(x));
}

/// \brief Solve in place on dense objects: x holds the unpermuted
/// right-hand side on entry and the solution on exit.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in]     A   factored matrix-like object
/// \param[in]     piv pivot storage produced by factorize
/// \param[in,out] x   vector-like right-hand side, then solution
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE void substitute_inplace(const MatrixType& A, const PivotType& piv,
                                                          SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute_inplace writes into x");
    ctx::solver::template substitute_inplace<xt::is_internal, pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), xt::pointer(x),
        xt::stride(x));
}

/// \brief Solve A x = e_col on dense objects, from a prior factorize -
/// the consistent-tangent-operator path.
/// \tparam UserConfig compile-time knobs (void = TiledLUppDefaultConfig)
/// \param[in]  A   factored matrix-like object
/// \param[in]  piv pivot storage produced by factorize
/// \param[in]  col index of the canonical column e_col
/// \param[out] x   vector-like solution
template<typename UserConfig = void, typename MatrixType, typename PivotType, typename SolutionType>
TDLS_HOST_DEVICE TDLS_FORCEINLINE void
substitute_canonical(const MatrixType& A, const PivotType& piv, const int col, SolutionType& x) {
    using ctx = detail::adaptor_context<UserConfig, MatrixType>;
    using mt  = typename ctx::mtraits;
    using pa  = detail::pivot_access<PivotType>;
    detail::check_vector<SolutionType, typename ctx::scalar, ctx::N>();
    using xt = storage_traits<std::remove_cv_t<SolutionType>>;
    static_assert(xt::is_mutable, "tdls adaptors: substitute_canonical writes into x");
    ctx::solver::template substitute_canonical<xt::is_internal, pa::is_internal, mt::is_internal>(
        mt::pointer(A), mt::stride(A), pa::pointer(piv), pa::stride(piv), col, xt::pointer(x),
        xt::stride(x));
}



} // namespace tdls



#endif // TDLS_ADAPTORS_HPP
