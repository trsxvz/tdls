# TDLS - Tiny Device-callable Linear Solvers

[![ci](https://github.com/trsxvz/tdls/actions/workflows/ci.yml/badge.svg)](https://github.com/trsxvz/tdls/actions/workflows/ci.yml)
[![docs](https://github.com/trsxvz/tdls/actions/workflows/docs.yml/badge.svg)](https://trsxvz.github.io/tdls/)

TDLS is a header-only C++17 library of direct solvers for small
general linear systems. It is written to be callable from device code:
one thread solves one system, on CPU as well as inside a CUDA, HIP,
SYCL, Kokkos, AdaptiveCpp, stdpar or OpenMP kernel. The library has no
dependency and no installation step.

The only solver family available today is TiledLUpp, an LU
factorization with logical partial pivoting on a tile grid. It comes
in two variants.
`TiledLUppSolverStatic` takes the dimension at compile time; residency
booleans let whole systems live in registers. `TiledLUppSolverDynamic`
takes the dimension at run time; the placement of the operands is
expressed through element strides.

```cpp
#include <tdls/tdls.hpp>

// Factorize once, reuse the factorization for every right-hand side.
using Solver = tdls::TiledLUppSolverStatic<double, 9, tdls::TiledLUppConfig<double, 3>>;
Solver::factorize<true, true>(M, 1, piv, 1);
Solver::substitute<true, true, true>(M, 1, piv, 1, r, dz, 1);
```

Dense math objects can also be passed directly: the adaptors of
`tdls/adaptors.hpp` recognize matrices, vectors and strided views
structurally, without naming any external library.

## Using the library

Copy the `include/` directory into a project, or use CMake:

```cmake
add_subdirectory(tdls)
target_link_libraries(my_target PRIVATE tdls::tdls)
```

## Tests and examples

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -L solvers    # library test suites
ctest --test-dir build -L examples   # self-checking examples
```

The examples cover two scientific problems at four execution scales:
sequential, OpenMP, and two GPU placements. The GPU examples build
when a CUDA or HIP toolchain is found.

## Documentation

The documentation, including the Doxygen API reference, lives at
<https://trsxvz.github.io/tdls/>.

## License

TDLS is designed to be embedded in
[TFEL/MFront](https://github.com/thelfer/tfel); its BSD-3-Clause
license places no meaningful restriction on such use.
