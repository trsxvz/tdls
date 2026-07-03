# Tests

The test tree is split into separated spaces, each tagged with a ctest
label so they run independently: `solvers` validates the library
itself, `examples` runs the self-checking examples (see
{doc}`examples`), and `benchmarks` will validate the benchmark harness
once that part exists.

```sh
ctest --test-dir build -L solvers      # one space
ctest --test-dir build -R oracle       # filter suites by name
./build/tests/solvers/tdls_test_tiledlupp_oracle_static N=12
```

The last form runs a single suite directly; the optional argument
keeps only the cases whose name contains it. The suites use a small
self-contained harness (`tests/common/harness.hpp`), no external test
framework.

## Architecture

The `solvers` space rests on two pillars.

The anchor is an independent oracle: a naive textbook LU with physical
row swaps (`tests/common/reference_lu.hpp`), written against none of
the library internals. Solver results are compared to it through the
normwise backward error, the conditioning-independent measure of a
direct solve. The tolerances are derived from the pivoting policy of
the configuration, not guessed.

Everything else is bitwise bridging: once one path is anchored, every
other path must reproduce it bit for bit. The runtime variant against
the compile-time one, every residency combination, every batch layout,
every entry point equivalence, every tile size. The dimension grid of
the anchor suites crosses every tile boundary (divisible, off by one,
single tile, tile equal to the dimension), so a bridge failure
localizes a divergence exactly.

## The suites

| Suite | What it locks |
|---|---|
| `oracle_static`, `oracle_dynamic` | backward error of every entry point against the naive LU, on the boundary-crossing dimension grid |
| `static_vs_dynamic` | bitwise equality of the two variants |
| `residencies` | every residency combination reproduces the anchored path bitwise |
| `layouts` | AoS, SoA and AoSoA addressing, bitwise |
| `entry_points` | the documented entry point equivalences, bitwise |
| `inplace_paths` | the two in-place substitution algorithms around their switchover dimensions |
| `tile_sizes` | every tile size against the anchored one, plus the runtime TS > n cases |
| `singular` | singular and near-singular systems, including the tiny-but-solvable counter-case |
| `config_knobs` | each configuration knob changes what it should and nothing else |
| `adaptors` | structural detection on mock objects, accepted and rejected shapes |
| `cross_full` | full parameter cross-product on two rich shapes |
| `dynamic_edges` | runtime-size edge cases of the dynamic variant |
| `reject_config_*` | negative compilation tests of the configuration contracts |

The `reject_config_*` entries deserve a word: they compile, on
purpose, translation units that must be rejected by the compile-time
contracts of the configuration (scalar type of the thresholds, in both
directions, and threshold ordering). Each test passes only when the
compiler emits the exact diagnostic of its contract, so a silently
dropped contract turns the suite red.

For the detail of any suite, the authoritative description is the
`\file` documentation at the top of its source in
`tests/solvers/tiled_lupp/`.
