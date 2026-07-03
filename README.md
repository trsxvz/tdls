# TDLS - Tiny Device-callable Linear Solvers

[![ci](https://github.com/trsxvz/TDLS/actions/workflows/ci.yml/badge.svg)](https://github.com/trsxvz/TDLS/actions/workflows/ci.yml)
[![docs](https://github.com/trsxvz/TDLS/actions/workflows/docs.yml/badge.svg)](https://trsxvz.github.io/TDLS/)

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
#include <limits>

#include <tdls/tdls.hpp>

// Solver configuration, every knob spelled out.
// tdls::TiledLUppDefaultConfig<T> can instead provide ready-made defaults.
struct Config {
    // Size of the tiles used for the LU factorization
    static constexpr int tile_size = 3;
    // Elimination schedule, RightLooking or LeftLooking
    static constexpr tdls::TiledLUppSchedule sched = tdls::TiledLUppSchedule::RightLooking;
    // A pivot at least this large is accepted without searching outside the tile
    static constexpr double oot_threshold = 1e-10;
    // The factorization is declared singular when the best pivot falls below this floor
    static constexpr double singular_eps = std::numeric_limits<double>::min();
    // The out-of-tile search stops at the first acceptable pivot
    static constexpr bool oot_first_acceptable = true;
    // Forced unrolling of the in-tile loops
    static constexpr bool unroll_inner = true;
};

// LU solver for systems of dimension 9: the 9 x 9 matrix is cut
// into 3 x 3 register tiles. One call: factorize M in place, then
// overwrite y with the solution. The residency booleans declare
// every operand caller-local, so the stride arguments (the 1s) are
// ignored at compile time; with external operands they carry the
// element stride of each array.
using Solver = tdls::TiledLUppSolverStatic<double, 9, Config>;
Solver::solve_inplace<true, true, true>(M, 1, piv, 1, y, 1);
```

Dense math objects can also be passed directly: the adaptors of
`tdls/core/adaptors.hpp` recognize matrices, vectors and strided views
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
<https://trsxvz.github.io/TDLS/>.

## License

TDLS is designed to be embedded in
[TFEL/MFront](https://github.com/thelfer/tfel); its BSD-3-Clause
license places no meaningful restriction on such use.
