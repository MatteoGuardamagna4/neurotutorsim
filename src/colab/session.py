"""Colab session bootstrap, shared by every Track A notebook.

A notebook may hold bootstrap steps and one call into `scripts/`, nothing else
(§4.3): a cell dies with its session, so anything that *decides* something has
to live in the repo. Choosing which disk holds 13 GB of re-downloadable weights
and which holds predictions that cost GPU time against a gated model is a
decision, so it lives here -- and every notebook gets the same answer by
construction rather than by three copies of a cell staying in sync.

Track A entry point, but it imports neither torch nor tribev2. It sets
environment variables and makes directories, so its tests run on a laptop.

    from src.colab import setup_session
    paths = setup_session()

`cache_root=` redirects predictions somewhere other than the project's Drive
folder, which is the only thing a second inference notebook needs to differ in.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

#: Colab's runtime disk: ~100 GB, wiped when the session ends.
COLAB_LOCAL_ROOT = Path("/content")
#: Where `google.colab.drive` mounts, and the root it exposes.
COLAB_MOUNT_POINT = Path("/content/drive")
COLAB_DRIVE_ROOT = COLAB_MOUNT_POINT / "MyDrive"
COLAB_REPO_DIR = Path("/content/neurotutorsim")

DEFAULT_PROJECT = "NeuroTutorSim"


class SessionError(RuntimeError):
    """Raised when the session cannot be configured as asked."""


@dataclass(frozen=True)
class SessionPaths:
    """Every path a Track A notebook needs, derived once."""

    on_colab: bool
    repo_dir: Path
    drive_root: Path
    project_drive: Path
    cache_root: Path
    handoff: Path
    hf_cache: Path
    auth_status: str

    def summary(self) -> str:
        return "\n".join(
            [
                f"repo       : {self.repo_dir}  (exists={self.repo_dir.exists()})",
                f"cache root : {self.cache_root}  (Drive -- keep)",
                f"handoff    : {self.handoff}",
                f"HF cache   : {self.hf_cache}  (ephemeral -- re-downloaded each session)",
                f"auth       : {self.auth_status}",
            ]
        )


def setup_session(
    *,
    project: str = DEFAULT_PROJECT,
    repo_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    drive_root: str | Path | None = None,
    local_root: str | Path | None = None,
    authenticate: bool = True,
) -> SessionPaths:
    """Mount Drive, derive the session's paths, point `HF_HOME` at local disk.

    Safe to re-run: the same call twice is a no-op, which matters because this
    is the cell you re-run after a restart or a dropped connection.
    """
    colab_drive = _import_colab("drive")
    on_colab = colab_drive is not None

    resolved_drive = Path(drive_root) if drive_root else _default_drive_root(on_colab)
    resolved_local = Path(local_root) if local_root else _default_local_root(on_colab)
    resolved_repo = Path(repo_dir) if repo_dir else _default_repo_dir(on_colab)

    # Only mount when we are actually going to use the mount: an explicit
    # drive_root (tests, or a session already mounted) must not trigger one.
    if on_colab and drive_root is None:
        colab_drive.mount(str(COLAB_MOUNT_POINT))

    project_drive = resolved_drive / project
    paths = SessionPaths(
        on_colab=on_colab,
        repo_dir=resolved_repo,
        drive_root=resolved_drive,
        project_drive=project_drive,
        cache_root=Path(cache_root) if cache_root else project_drive / "tribe_cache",
        handoff=project_drive / "gate17",
        hf_cache=resolved_local / "hf_cache",
        auth_status="skipped",
    )

    # The storage decision, in mechanical form. ~13 GB of weights on a 15 GB
    # Drive leaves no room for the ~4.7 GB corpus they exist to serve, and Drive
    # cannot hold symlinks, so huggingface_hub stores every blob twice there.
    if paths.hf_cache.is_relative_to(paths.drive_root):
        raise SessionError(
            f"the HuggingFace cache would land at {paths.hf_cache}, inside the Drive root "
            f"{paths.drive_root}. Weights are ~13 GB and re-download in about a minute; "
            f"predictions cost GPU time against a gated model and cannot be re-fetched. "
            f"Pass a local_root outside Drive."
        )

    for directory in (paths.project_drive, paths.cache_root, paths.handoff, paths.hf_cache):
        directory.mkdir(parents=True, exist_ok=True)

    _configure_hf_home(paths.hf_cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Exported, not just assigned: the `!` cells launch subprocesses, which
    # inherit the environment but know nothing about the kernel's variables.
    os.environ["CACHE_ROOT"] = str(paths.cache_root)
    os.environ["REPO_DIR"] = str(paths.repo_dir)

    if authenticate:
        paths = _replace_auth(paths, _authenticate(on_colab))

    print(paths.summary())
    return paths


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _import_colab(name: str):
    """Return `google.colab.<name>`, or None off Colab."""
    try:
        module = __import__(f"google.colab.{name}", fromlist=[name])
    except ImportError:
        return None
    return module


def _default_drive_root(on_colab: bool) -> Path:
    return COLAB_DRIVE_ROOT if on_colab else Path("drive_local").resolve()


def _default_local_root(on_colab: bool) -> Path:
    return COLAB_LOCAL_ROOT if on_colab else Path.cwd()


def _default_repo_dir(on_colab: bool) -> Path:
    return COLAB_REPO_DIR if on_colab else Path(__file__).resolve().parent.parent.parent


def _replace_auth(paths: SessionPaths, status: str) -> SessionPaths:
    from dataclasses import replace

    return replace(paths, auth_status=status)


def _configure_hf_home(hf_cache: Path) -> None:
    """Point `HF_HOME` at `hf_cache`, or stop if it is already too late.

    huggingface_hub resolves its cache directory once, at import time. Setting
    `HF_HOME` afterwards is silently ignored -- the directory we just made would
    stay empty while several GB landed somewhere else. So a hub module that is
    already imported is only acceptable if it happens to agree with us, which
    makes re-running the setup cell a no-op instead of an error.
    """
    os.environ["HF_HOME"] = str(hf_cache)
    if "huggingface_hub" not in sys.modules:
        return

    from huggingface_hub import constants

    effective = Path(constants.HF_HUB_CACHE).resolve()
    wanted = (hf_cache / "hub").resolve()
    if effective != wanted:
        raise SessionError(
            f"huggingface_hub is already imported in this process with its cache at "
            f"{effective}, and that is fixed at import time -- setting HF_HOME now would be "
            f"ignored, and several GB would land there instead of {wanted}. "
            f"Runtime -> Restart session, then run this cell before anything else."
        )


def _authenticate(on_colab: bool) -> str:
    """Log into HuggingFace. Imported here, after `HF_HOME` is set.

    `meta-llama/Llama-3.2-3B` (TRIBE's text encoder) is gated per account, so a
    token is not optional on Track A -- but Track B laptops never need one, and
    a missing token there is reported rather than raised.
    """
    if on_colab:
        userdata = _import_colab("userdata")
        try:
            token = userdata.get("HF_TOKEN")
        except Exception as exc:
            raise SessionError(
                f"could not read HF_TOKEN from Colab Secrets ({type(exc).__name__}: {exc}). "
                f"Open the key icon in the sidebar, add HF_TOKEN, and toggle notebook access "
                f"on for this notebook. Secrets are per Google account, so a new account "
                f"needs it added again."
            ) from exc
        source = "Colab Secrets"
    else:
        token = os.environ.get("HF_TOKEN")
        source = "HF_TOKEN environment variable"

    if not token:
        return f"not authenticated (no token in {source})"

    from huggingface_hub import login

    login(token=token)
    return f"authenticated from {source}"
