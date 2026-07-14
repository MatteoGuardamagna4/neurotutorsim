"""Package-structure gate.

Importing every subpackage of ``src`` guards the layout declared in
``[tool.hatch.build.targets.wheel] packages = ["src"]``. If someone breaks the
skeleton (renames a folder, drops an ``__init__.py``), these tests fail loudly
instead of surfacing as a cryptic ``ModuleNotFoundError`` deep in Colab.
"""

import importlib

import pytest

SUBMODULES = [
    "src.generation",
    "src.validation",
    "src.tribe",
    "src.learners",
    "src.plasticity",
    "src.analysis",
    "src.visualization",
]


@pytest.mark.parametrize("module", SUBMODULES)
def test_submodule_imports(module):
    assert importlib.import_module(module) is not None
