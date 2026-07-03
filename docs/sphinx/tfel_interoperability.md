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

The detection is purely structural: tdls names and includes no
external library. Two shapes are recognized:

- contiguous fixed-size objects and views, exposing `data()` and an
  `indexing_policy` type with constexpr extents (this matches
  `tfel::math` `tmatrix`, `tvector` and `View`, among others);
- element-strided views, exposing their runtime stride either through
  a `data()` member returning a (pointer, stride) pair (this matches
  `tfel::math` `StridedCoalescedView`) or through a `stride()` /
  `getStride()` member next to a plain `data()`.

Residencies are inferred per argument (contiguous object: internal;
strided view: external) and can be mixed freely in one call. Two
shapes are rejected at compile time with an explicit message:
row-strided sub-matrix views, whose row stride cannot be expressed by
the single-stride addressing, and gather views, which expose no
`data()` at all.

For exotic types the detection can be overridden by specializing
`tdls::storage_traits`.

The (pointer, stride) form of `StridedCoalescedView::data()` was
upstreamed to TFEL and is part of TFEL master.
