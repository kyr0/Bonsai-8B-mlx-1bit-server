# autokernel/Makefile — experiment loop for MLX kernel optimization
#
# Usage:
#   make baseline          # establish baseline tok/s
#   make experiment ID=1 TARGET=quantized_dispatch_decode DESC="tile size 2x"
#   make compile           # rebuild MLX from ../mlx
#   make revert            # restore MLX source to git HEAD
#   make status            # show manifest summary
#   make analyze           # generate analysis report

SHELL       := /bin/zsh
VENV        := ../.venv
PYTHON      := $(VENV)/bin/python
MLX_REPO    := ../mlx
MODEL       := prism-ml/Bonsai-8B-mlx-1bit
TARGET      := quantized_dispatch_decode
ID          ?=
DESC        ?=

.PHONY: baseline experiment compile revert status analyze test-patch help

help:
	@echo "AutoKernel — MLX kernel optimization experiment loop"
	@echo ""
	@echo "Targets:"
	@echo "  baseline                  Establish baseline performance (experiment 0)"
	@echo "  experiment                Run an experiment (requires ID= TARGET= DESC=)"
	@echo "  compile                   Rebuild and install MLX from source"
	@echo "  revert                    Revert MLX source tree to git HEAD"
	@echo "  status                    Print manifest summary"
	@echo "  analyze                   Generate analysis report"
	@echo "  test-patch                Round-trip patch test (no-op experiment)"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL   = $(MODEL)"
	@echo "  TARGET  = $(TARGET)"
	@echo "  MLX_REPO = $(MLX_REPO)"
	@echo ""
	@echo "Examples:"
	@echo "  make baseline"
	@echo "  make experiment ID=1 TARGET=sdpa_vector_decode DESC='doubled BN'"
	@echo "  make status"

# -- baseline -----------------------------------------------------------

baseline:
	@echo "=> Establishing baseline ..."
	$(PYTHON) -m autokernel.src.orchestrate baseline \
		--model $(MODEL) \
		--target $(TARGET) \
		--mlx-repo $(MLX_REPO)

# -- experiment ---------------------------------------------------------

experiment:
	@if [ -z "$(ID)" ]; then \
		echo "ERROR: ID is required. Usage: make experiment ID=1 TARGET=... DESC='...'"; \
		exit 1; \
	fi
	@if [ -z "$(DESC)" ]; then \
		echo "ERROR: DESC is required. Usage: make experiment ID=1 DESC='description'"; \
		exit 1; \
	fi
	$(PYTHON) -m autokernel.src.orchestrate experiment \
		--target $(TARGET) \
		--candidate-dir candidates/$(ID) \
		--experiment-id $(ID) \
		--description "$(DESC)" \
		--model $(MODEL) \
		--mlx-repo $(MLX_REPO)

# -- compile ------------------------------------------------------------

compile:
	@echo "=> Compiling MLX from $(MLX_REPO) ..."
	$(PYTHON) -m autokernel.src.compile_mlx --mlx-repo $(MLX_REPO)

# -- revert -------------------------------------------------------------

revert:
	@echo "=> Reverting MLX source tree ..."
	$(PYTHON) -m autokernel.src.patch revert --mlx-repo $(MLX_REPO) --target $(TARGET)
	@echo "=> Done."

# -- status -------------------------------------------------------------

status:
	@$(PYTHON) -m autokernel.src.orchestrate status

# -- analyze ------------------------------------------------------------

analyze:
	@$(PYTHON) -m autokernel.src.analysis

# -- test-patch ---------------------------------------------------------
# Run a no-op experiment: copy baseline kernels as "candidate", run the
# full loop.  Verifies the patch/compile/benchmark pipeline works.

test-patch:
	@echo "=> Running no-op experiment (round-trip test) ..."
	@mkdir -p candidates/test_noop
	@# Copy baseline kernel files as the "candidate"
	$(PYTHON) -c "\
	from autokernel.src.manifest import get_target; \
	from pathlib import Path; \
	import shutil; \
	t = get_target('$(TARGET)'); \
	mlx = Path('$(MLX_REPO)').resolve(); \
	dest = Path('candidates/test_noop'); \
	[(lambda d: (d.parent.mkdir(parents=True, exist_ok=True), shutil.copy2(str(mlx/f), str(d))))(dest/f) for f in t.source_files if (mlx/f).exists()]; \
	print(f'Copied {len(t.source_files)} files')"
	$(PYTHON) -m autokernel.src.orchestrate experiment \
		--target $(TARGET) \
		--candidate-dir candidates/test_noop \
		--experiment-id 999 \
		--description "NO-OP round-trip test" \
		--model $(MODEL) \
		--mlx-repo $(MLX_REPO) \
		--skip-compile
	@echo "=> Test complete. Check candidates/999/result.json"
