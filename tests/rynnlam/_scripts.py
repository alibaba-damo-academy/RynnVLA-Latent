"""Load the RynnLAM entry points, which live in scripts/ rather than in the rynnlam package.

``scripts/`` is not an importable package (pyproject excludes it), and these files are meant to
be run, not imported. The upstream RynnLAM tests did ``import train`` against a repository whose
entry points sat at the top level; this release moved them under ``scripts/`` and renamed them,
so load them by path instead of re-arranging the tree to suit the tests.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def load_script(name, filename=None):
    """Import scripts/<filename> under a stable module name and return the module.

    Cached in sys.modules: these entry points pull in cv2/torch and run any module-level setup
    on import, so a second caller in the same session should not pay for it again.
    """
    # Running `python scripts/x.py` puts scripts/ on sys.path[0], which is how the entry points
    # import each other. Loading by path does not, so reproduce it -- without this, loading
    # evaluate_rynnlam.py fails on its `import stream_extract_tar`.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    module_name = f"_rynnlam_script_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = SCRIPTS / (filename or f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a dataclass or a self-reference inside the script resolves.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
