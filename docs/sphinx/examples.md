# Examples

The examples double as living documentation and as tests: each one is
a self-checking program executed by ctest under the `examples` label.
They are organized as two scientific problems, each declined at every
execution scale, so that moving from one column to the next changes
only the parallel harness, never the solver calls.

| | CPU, sequential | CPU, parallel | GPU, thread-local memory | GPU, global device memory |
|---|---|---|---|---|
| **Compile-time dimension** (stiff chemistry, Radau IIA, N = 9) | `implicit_ode` | `implicit_ode_batch_omp` | `implicit_ode_batch_gpu` | `implicit_ode_batch_gpu_soa` |
| **Runtime dimension** (Love integral equation, Nystroem, n chosen at launch) | `integral_equation` | `integral_equation_batch_omp` | `integral_equation_batch_gpu` | `integral_equation_batch_gpu_soa` |

The sources live under `examples/tiled_lupp/`, split by execution
scale (`cpu_sequential/`, `cpu_openmp/`, `gpu_cuda_or_hip/`). The GPU
examples are single sources in the common CUDA/HIP dialect, compiled
by whichever toolchain the configure step finds.

## The compile-time family

The Robertson stiff kinetics is integrated with a 3-stage Radau IIA
method: the Newton systems have size N = stages x species, fixed by
the method and by the chemical mechanism, so N = 9 is a property of
the program and `TiledLUppSolverStatic` applies. The examples show the
canonical reason for the split factorize/substitute interface (frozen
Jacobian practice) and, on GPU, the two residency choices:

- `implicit_ode_batch_gpu`: residency booleans set to true, the whole
  system lives in thread registers, device memory only holds the
  per-cell inputs and outputs;
- `implicit_ode_batch_gpu_soa`: residency booleans set to false, the
  batch of systems is materialized in device memory
  structure-of-arrays and walked with the batch stride, every access
  coalesced.

## The runtime family

Love's integral equation (the potential of a parallel-plate
capacitor) is discretized with the Nystroem method: the quadrature
resolution n is an accuracy versus cost knob chosen when the
computation is launched, so the dimension is a runtime value and
`TiledLUppSolverDynamic` applies. The batch versions sweep the plate
separation, one dense system per parameter value. The runtime solver
has no residency booleans; the two GPU variants express the placement
through the strides handed to the solver (unit stride on thread-local
arrays, batch stride on the SoA batch).

Every instance factorizes once and substitutes two right-hand sides:
a manufactured one, which the solve must return to solver accuracy
and serves as the self-check, and the physical unit potential.
