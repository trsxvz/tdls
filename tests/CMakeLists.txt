# The test tree is split by target space, each tagging its ctest
# entries with a label so they run independently (ctest -L <label>):
# tests/solvers validates the library itself ("solvers"); the examples
# register themselves as self-checking tests from examples/
# ("examples"); tests/benchmarks will validate the benchmark harness
# once that part exists.
add_subdirectory(solvers)
