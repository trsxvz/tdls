# Getting started

## Requirements

A C++17 compiler. Nothing else: the library has no dependency and is
tested against g++, clang++, nvcc and hipcc.

## Consuming the library

tdls is header-only and has no installation step. Either copy the
`include/` directory into a project and add it to the include path, or
use CMake:

```cmake
add_subdirectory(tdls)
target_link_libraries(my_target PRIVATE tdls::tdls)
```

Then include the umbrella header:

```cpp
#include <tdls/tdls.hpp>
```

Finer-grained headers exist for the individual pieces
(`tdls/solvers/tiled_lupp/solver_static.hpp`,
`tdls/solvers/tiled_lupp/solver_dynamic.hpp`, `tdls/adaptors.hpp`).

## Building the tests and examples

Both are ordinary CMake targets, enabled by default when tdls is the
top-level project and disabled when it is consumed through
`add_subdirectory` (options `TDLS_BUILD_TESTS` and
`TDLS_BUILD_EXAMPLES`):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -L solvers    # library test suites
ctest --test-dir build -L examples   # self-checking examples
```

The OpenMP examples build when an OpenMP runtime is found; the GPU
examples build when a CUDA or HIP toolchain is found and report
themselves as skipped when no device is present.

## Portability model

The headers decorate every entry point with a small set of macros
(`TDLS_HOST_DEVICE`, `TDLS_FORCEINLINE`, `TDLS_RESTRICT`,
`TDLS_UNROLL_FORCE`) that resolve to the right annotation for the
compiler at hand: `__host__ __device__` under CUDA and HIP, plain
host code elsewhere. Single-source models (SYCL, stdpar, OpenMP
target, Kokkos through its backend compiler) need no decoration at
all. Every macro is `#ifndef`-guarded, so any of them can be
overridden from the command line without editing the headers.
