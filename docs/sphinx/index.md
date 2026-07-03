# TDLS - Tiny Device-callable Linear Solvers

TDLS is a header-only C++17 library of direct solvers for small
general linear systems, written to be callable from device code: one thread
solves one system, on CPU as well as inside a CUDA, HIP, SYCL, Kokkos,
AdaptiveCpp, stdpar or OpenMP (host and offload) kernel. It has no
dependency and no installation step.

The only solver family available today is TiledLUpp, an LU
factorization with logical partial pivoting operating on a tile grid,
in two variants:

- `tdls::TiledLUppSolverStatic<T, N, Config>`: the dimension N is a
  compile-time constant. Residency template booleans let the operands
  live in registers (compile-time indexing) or in external memory
  (strided accesses).
- `tdls::TiledLUppSolverDynamic<T, Config>`: the dimension n is a
  runtime value. The placement of the operands is expressed through
  runtime element strides.

Both variants address every operand through a single element stride.
This covers the three batch layouts alike: AoS (contiguous storage,
stride 1), SoA (stride = batch size) and AoSoA (tiled hybrid); the SoA
and AoSoA layouts both provide memory coalescence.

A taste of the interface, from the examples:

```cpp
#include <tdls/tdls.hpp>

using StaticSolver  = tdls::TiledLUppSolverStatic<double, 9, tdls::TiledLUppConfig<double, 3>>;
using DynamicSolver = tdls::TiledLUppSolverDynamic<double>;

// Split interface: factorize once, then reuse the factorization for
// every right-hand side (Newton iterations, parameter sweeps, ...).
StaticSolver::factorize<true, true>(M, 1, piv, 1);
StaticSolver::substitute<true, true, true>(M, 1, piv, 1, r, dz, 1);

// Combined interface: solve() chains factorize() and substitute().
// Both variants offer both interfaces; the split is shown on the
// static solver and the combined one on the dynamic solver.
DynamicSolver::solve(n, A.data(), 1, piv.data(), 1, b.data(), x.data(), 1);
```

Dense math objects (matrices, vectors, strided views) can also be
passed directly through the structural adaptors, without naming any
external library; see {doc}`tfel_interoperability`.

```{toctree}
:maxdepth: 1

getting_started
examples
tfel_interoperability
api_reference
```
