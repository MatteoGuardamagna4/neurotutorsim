"""Track B: the synthetic learner engine (deliverable D4, brief §7).

Deliberately empty of re-exports. `tests/test_imports.py` imports this package,
and a laptop `uv sync` installs no torch -- so keeping this module free of
submodule imports means importing the package cannot pull in anything heavy, and
the CPU-purity rule in `CLAUDE.md` is checked against the modules themselves
rather than against whatever this file happens to re-export.

Import the modules directly:

    from src.learners.config import load_learner_config
    from src.learners.curriculum import load_curriculum
    from src.learners.simulate import run_simulation
"""
