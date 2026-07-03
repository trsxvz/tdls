# TFEL interoperability

The free functions of `tdls/core/adaptors.hpp` accept dense math objects
directly and infer everything the raw interface needs: the scalar
type, the dimension, the strides and the residency booleans.

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

Plain `TFEL` objects, then `TFEL` strided-coalesced views, in action:

```cpp
#include <TFEL/Math/Array/StridedCoalescedView.hxx>
#include <TFEL/Math/tmatrix.hxx>
#include <TFEL/Math/tvector.hxx>

#include <tdls/tdls.hpp>

// Plain TFEL objects. No alias and no configuration are needed: the
// adaptors are free functions that deduce the scalar type, the
// dimension, the strides and the residencies from their arguments,
// and use the default TiledLUpp configuration.
tfel::math::tmatrix<12, 12, double> A = ...;
tfel::math::tvector<12, double> y = ...; // right-hand side, overwritten
tfel::math::tvector<12, int> piv;
tdls::solve_inplace(A, piv, y);

// Strided SoA views: system number s of a batch of `stride`
// interleaved systems, mapped with tfel::math::map_strided. The
// strides and the residency booleans of the raw interface never
// appear in either call: the adaptors read the stride carried by
// each argument and infer its residency (plain object: internal;
// strided view: external). The pivot stays the plain local vector
// declared above: placements mix freely.
auto As = tfel::math::map_strided<tfel::math::tmatrix<12, 12, double>>(a_base + s, stride);
auto ys = tfel::math::map_strided<tfel::math::tvector<12, double>>(y_base + s, stride);
tdls::solve_inplace(As, piv, ys);
```
