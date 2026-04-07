# MLX AutoKernel Handoff Bundle

Contents:

- `VALIDATED_IMPLEMENTATION_PLAN.md` — validated plan and corrected hotspot ranking
- `autokernel_mlx/manifest.py` — validated target manifest
- `autokernel_mlx/backend.py` — backend adapter scaffold
- `autokernel_mlx/build_mlx.py` — MLX build / rebuild helper
- `autokernel_mlx/model_shapes.py` — decode / prefill shape presets and generators
- `autokernel_mlx/bench_mlx.py` — public-API benchmark harness
- `autokernel_mlx/results.py` — TSV result helpers

## Intended use

1. Copy `autokernel_mlx/` into the AutoKernel repo.
2. Keep MLX as a checked-out sibling or submodule.
3. Point `build_mlx.py` at that MLX checkout.
4. Run `bench_mlx.py` against the public MLX Python API.
5. Let another agent continue from there.

## Status

This package is source-validated against the uploaded MLX Metal source tree. It has not been built or executed in this container.
