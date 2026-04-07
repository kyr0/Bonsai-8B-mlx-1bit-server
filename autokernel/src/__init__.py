"""MLX backend scaffolding for an AutoKernel port."""

from .backend import MLXBackendAdapter, MLXBuildConfig
from .manifest import TARGETS, get_target, list_targets
from .model_shapes import TextModelShape, get_builtin_shape
from .patch import apply_patch, revert_to_baseline, create_patch
from .compile_mlx import compile_mlx
from .manifest_manager import load_manifest, add_experiment, get_best, summary
