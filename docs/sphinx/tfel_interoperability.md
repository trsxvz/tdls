# TFEL interoperability

The free functions of `tdls/adaptors.hpp` accept dense math objects
directly and infer everything the raw interface needs: the scalar
type, the dimension, the strides and the residency booleans.

```cpp
#include <tdls/tdls.hpp>

tfel::math::tmatrix<12, 12, double> A = ...;
tfel::math::tvector<12, double> b = ..., x;
tfel::math::tvector<12, int> piv;
tdls::solve(A, piv, b, x);
```

The detection is purely structural: TDLS names and includes no
external library. Two shapes are recognized:

- contiguous fixed-size objects and views, exposing `data()` and an
  `indexing_policy` type with constexpr extents (this matches
  `tfel::math::tmatrix`, `tfel::math::tvector` and `tfel::math::View`,
  among others);
- element-strided views, exposing their runtime stride either through
  a `data()` member returning a (pointer, stride) pair (this matches
  `tfel::math::StridedCoalescedView`) or through a `stride()` /
  `getStride()` member next to a plain `data()`.

The elements of a `tfel::math::ViewsArray` (the result of `map_array`)
are themselves `View` objects and are accepted individually.

Residencies are inferred per argument (contiguous object: internal;
strided view: external) and can be mixed freely in one call. Two
shapes are rejected at compile time with an explicit message:
row-strided sub-matrix views (such as the result of
`tfel::math::map_derivative()`, which maps a derivative block inside a
larger jacobian matrix, or of `tfel::math::tmatrix::submatrix_view()`),
whose row stride cannot be expressed by the single-stride addressing,
and gather views holding one pointer per component (this matches
`tfel::math::CoalescedView`, built by the `map()` overload taking an
array of pointers), which expose no `data()` at all.

For exotic types the detection can be overridden by specializing
`tdls::storage_traits`.
