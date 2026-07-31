"""Track B stays CPU-pure: nothing under `src/learners/` may import torch.

The reason is architectural, not stylistic. The learner engine runs millions of
times on a laptop; the Minitaur probe runs once on a Colab GPU. If the two ever
become coupled, the cheap path depends on the expensive one and a laptop
`uv sync` -- which installs no torch -- stops being able to run Phase III at all.

Testing this honestly on *this* machine takes some care. The developer `.venv`
here has torch 2.6.0 installed, because `uv sync --extra tribe` was run for Track
A. So "import it in an environment with no torch" cannot be done literally. The
two tests below are stronger than the literal version anyway:

* a static scan of every module for a forbidden import, at any nesting depth,
  which catches an import inside a function that a runtime check would miss
  unless that function happened to be called; and
* a subprocess import with the forbidden modules made unimportable by a meta-path
  blocker, which reproduces the laptop environment deterministically rather than
  depending on what happens to be installed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

LEARNERS = PROJECT_ROOT / "src" / "learners"

#: Rule 2 of the Phase III task spec.
FORBIDDEN_PACKAGES = ("torch", "transformers", "unsloth", "bitsandbytes", "accelerate")

#: Modules a laptop must be able to import with none of the above present.
LEARNER_MODULES = (
    "src.learners",
    "src.learners.config",
    "src.learners.seeds",
    "src.learners.curriculum",
    "src.learners.population",
    "src.learners.responses",
    "src.learners.engine",
    "src.learners.support",
    "src.learners.effort",
    "src.learners.updates",
    "src.learners.episode",
    "src.learners.checkpoints",
    "src.learners.simulate",
    "src.learners.minitaur_prompt",
)


def _module_paths() -> list[Path]:
    return sorted(LEARNERS.glob("*.py"))


def test_the_module_list_is_complete():
    """A new module must be added here, or it escapes both checks below."""
    on_disk = {f"src.learners.{path.stem}" for path in _module_paths()}
    on_disk.discard("src.learners.__init__")
    listed = set(LEARNER_MODULES) - {"src.learners"}
    assert on_disk == listed, f"unlisted module(s): {sorted(on_disk ^ listed)}"


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_no_forbidden_import_at_any_depth(path: Path):
    """Catches a lazy `import torch` inside a function, not just a top-level one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] in FORBIDDEN_PACKAGES:
                offenders.append(f"line {node.lineno}: {name}")
    assert offenders == [], f"{path.name} imports a GPU package: {offenders}"


@pytest.mark.parametrize("path", _module_paths(), ids=lambda p: p.name)
def test_no_global_numpy_random_call(path: Path):
    """§10.1: every draw comes from a named substream, never the global state.

    `np.random.default_rng`, `Generator`, `SeedSequence` and `PCG64` are
    constructors, not draws off the process-wide state, so they are allowed. A
    single `np.random.normal(...)` anywhere would reorder every draw downstream
    and break bitwise reproducibility.
    """
    allowed = {"default_rng", "Generator", "SeedSequence", "PCG64", "BitGenerator"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "random"
            and isinstance(value.value, ast.Name)
            and value.value.id in {"np", "numpy"}
            and node.attr not in allowed
        ):
            offenders.append(f"line {node.lineno}: np.random.{node.attr}")
    assert offenders == [], f"{path.name} uses the global numpy RNG: {offenders}"


def test_modules_import_with_every_gpu_package_blocked(tmp_path):
    """The gate-8.3 check, made deterministic.

    Installs a meta-path finder that raises `ImportError` for every forbidden
    package, then imports the whole of `src/learners/`. This reproduces a laptop
    `uv sync` regardless of what is installed in the developer's venv -- which on
    this machine includes torch, from the Track A `tribe` extra.
    """
    script = tmp_path / "blocked_import.py"
    script.write_text(
        "\n".join(
            [
                "import importlib, sys",
                f"FORBIDDEN = {set(FORBIDDEN_PACKAGES)!r}",
                "class Blocker:",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname.split('.')[0] in FORBIDDEN:",
                "            raise ImportError(f'blocked by the Track B purity test: {fullname}')",
                "        return None",
                "sys.meta_path.insert(0, Blocker())",
                f"for name in {list(LEARNER_MODULES)!r}:",
                "    importlib.import_module(name)",
                "print('OK')",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout


def test_the_blocker_itself_works(tmp_path):
    """Guards against the previous test passing because the blocker does nothing."""
    script = tmp_path / "blocker_selfcheck.py"
    script.write_text(
        "\n".join(
            [
                "import importlib, sys",
                "class Blocker:",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname.split('.')[0] == 'torch':",
                "            raise ImportError('blocked')",
                "        return None",
                "sys.meta_path.insert(0, Blocker())",
                "try:",
                "    importlib.import_module('torch')",
                "except ImportError:",
                "    print('BLOCKED')",
                "else:",
                "    print('NOT BLOCKED')",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert "BLOCKED" in result.stdout


def test_the_probe_script_imports_only_minitaur_prompt():
    """§6.2: the Colab probe may import exactly one Track B module."""
    tree = ast.parse(
        (PROJECT_ROOT / "scripts" / "benchmark_minitaur.py").read_text(encoding="utf-8")
    )
    learner_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.learners"):
            learner_imports.add(node.module)
        elif isinstance(node, ast.Import):
            learner_imports.update(
                alias.name for alias in node.names if alias.name.startswith("src.learners")
            )
    assert learner_imports == {"src.learners.minitaur_prompt"}, learner_imports


def test_the_probe_script_imports_gpu_packages_lazily():
    """torch must not be imported at module level, or `--preflight-only` needs a GPU box."""
    tree = ast.parse(
        (PROJECT_ROOT / "scripts" / "benchmark_minitaur.py").read_text(encoding="utf-8")
    )
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert not set(top_level) & set(FORBIDDEN_PACKAGES), top_level


def test_no_learner_module_reads_a_tribe_artefact():
    """The two tracks are parallel. Phase III computes no neural quantity."""
    offenders = []
    for path in _module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = " ".join(alias.name for alias in node.names)
            if module and ("src.tribe" in module or "src.analysis" in module):
                offenders.append(f"{path.name}:{node.lineno}: {module}")
    assert offenders == [], f"Track A/analysis import(s) in Track B: {offenders}"
