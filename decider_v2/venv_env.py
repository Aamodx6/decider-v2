"""venv_env.py -- a PRIVATE interpreter for the notebook's local subprocesses.

Why this exists
---------------
The notebook kernel orchestrates; it is never the pinned interpreter.  On a SageMaker Studio image
that is not merely a version mismatch -- it can be a combination that fails AT IMPORT TIME.  The
image this was written against carries `datasets 2.2.1` with `pyarrow 21`, and

    from datasets import load_dataset
    AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'

because `PyExtensionType` was removed in pyarrow 15.  No amount of version *checking* catches that,
because `importlib.metadata.version("pyarrow")` answers happily while the import explodes.

And `sys.executable` does not escape it: `subprocess.check_call([sys.executable, "prepare.py", ...])`
runs prepare.py on the SAME broken interpreter, so the carve fails even though the notebook "only
orchestrates".

The obvious fix -- `pip install -r requirements.txt` into the kernel -- is forbidden here, and rightly
so (see cell 2-install): upgrading numpy 2.3.5 / pyarrow 25 underneath every already-imported module
kills a long-lived Studio instance that other work depends on.

So the pins go in a THROWAWAY venv under the notebook's work directory:

    decider_v2_work/venv/            created with --system-site-packages
                                      so torch comes from the instance (never pip-installed),
                                      and only the pinned pure-binary set is overlaid.

The kernel is untouched.  Every local subprocess that needs the pins (prepare.py, tests.py, bench.py,
evaluate.py, calibrate.py) is invoked with this interpreter instead of `sys.executable`.

`--system-site-packages` is what makes this safe on a GPU instance: torch is deliberately unpinned
(environment.py UNPINNED) precisely because the container owns the CUDA-matched build, so the venv must
be able to SEE the instance's torch rather than shadow it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
STAMP = "pins.json"


def _venv_python(root: str) -> str:
    """The interpreter inside `root`, whether or not it exists yet."""
    if os.name == "nt":
        return os.path.join(root, "Scripts", "python.exe")
    return os.path.join(root, "bin", "python")


def _stamp_path(root: str) -> str:
    return os.path.join(root, STAMP)


def _read_stamp(root: str):
    try:
        with open(_stamp_path(root), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _requirements_fingerprint(requirements: str):
    """Hash of the pin file, so editing requirements.txt invalidates a stale venv."""
    import hashlib

    with open(requirements, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _pip_install(python: str, requirements: str) -> None:
    # --upgrade-strategy only: pip must replace the system-site-packages versions it shadows.  A plain
    # `pip install -r` reports "already satisfied" against the parent environment's copies and installs
    # nothing, leaving the venv just as broken as the kernel it was built to escape.
    cmd = [python, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
           "-r", requirements]
    print("venv: " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"pip install -r {requirements} failed with exit {rc}")


def _library_dirs() -> list:
    """Directories the PARENT interpreter needs to find its own shared libraries.

    This exists because of a failure that looks like a venv bug but is not one. On a conda-based image
    the kernel can `import torch` because conda's activation put the prefix's lib directory on the
    dynamic-loader search path. A venv subprocess launched directly never gets that activation, so
    torch -- which lives in the conda prefix, not the venv -- fails at import with

        OSError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory

    Note this happens while ctypes.CDLL loads libtorch_global_deps.so, i.e. BEFORE any of our code
    runs, so nothing inside the child can fix it. The fix has to be applied to the environment we
    hand the child. Recorded per-platform because the variable name differs.
    """
    dirs = []
    if os.name == "nt":
        # Windows resolves DLLs by directory search, not LD_LIBRARY_PATH; conda puts them next to the
        # interpreter (Library\\bin). Derive it from the parent's own prefix.
        base = sys.prefix
        for sub in (os.path.join("Library", "bin"), "DLLs"):
            p = os.path.join(base, sub)
            if os.path.isdir(p):
                dirs.append(p)
    else:
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            for p in os.environ.get(var, "").split(os.pathsep):
                if p and os.path.isdir(p):
                    dirs.append(p)
        # The parent prefix's own lib dir is the one conda activation would have added.
        for sub in ("lib", "lib64", os.path.join("lib", "python%d.%d" % sys.version_info[:2])):
            p = os.path.join(sys.base_prefix, sub)
            if os.path.isdir(p) and os.path.abspath(p) != os.path.abspath(sys.prefix):
                dirs.append(p)
    # De-duplicate, preserve order, keep existing entries first so we never demote what already works.
    seen, out = set(), []
    for p in dirs:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def patch_env(directories=None, prefix: str = "VENV") -> dict:
    """Add the parent's library directories to THIS process's environment, in place.

    Called from the notebook so that every later `subprocess` inherits the fix automatically -- the
    alternative is threading `env=...` through a dozen call sites and forgetting one.

    Returns the environment that was patched, for callers that pass it explicitly.
    """
    dirs = _library_dirs() if directories is None else list(directories)
    if not dirs:
        return dict(os.environ)
    if os.name == "nt":
        # PATH is what Windows searches for dependent DLLs.
        cur = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join(dirs + [p for p in cur if p])
    else:
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        cur = os.environ.get(var, "").split(os.pathsep)
        os.environ[var] = os.pathsep.join(dirs + [p for p in cur if p])
    print(f"{prefix}: added {len(dirs)} shared-library dir(s) to the child search path")
    return dict(os.environ)


def ensure(root: str, requirements: str = REQUIREMENTS, prefix: str = "VENV",
           force: bool = False) -> str:
    """Return the path to a pinned interpreter under `root`, creating/repairing the venv as needed.

    Idempotent: a venv whose stamp matches the current requirements.txt is reused untouched, so this
    costs a few milliseconds on every run after the first.
    """
    python = _venv_python(root)
    stamp = _read_stamp(root)
    want = _requirements_fingerprint(requirements)

    if not force and stamp and stamp.get("requirements_sha256") == want and os.path.exists(python):
        print(f"{prefix}: reusing {root} (pins unchanged)")
        return python

    if force and os.path.isdir(root):
        print(f"{prefix}: --force, rebuilding {root}")
        # Deliberately not shutil.rmtree: on Windows a stale venv can hold file handles and rmtree
        # leaves a half-deleted tree that `venv.EnvBuilder` then refuses to populate. Recreating over
        # the top of it with upgrade_deps is the supported path.
    print(f"{prefix}: building {root} (system site packages visible so torch stays the instance's)")
    venv.EnvBuilder(system_site_packages=True, with_pip=True, clear=False,
                    upgrade_deps=False).create(root)
    _pip_install(python, requirements)

    with open(_stamp_path(root), "w", encoding="utf-8") as fh:
        json.dump({"requirements": os.path.abspath(requirements),
                   "requirements_sha256": want,
                   "created_by_python": sys.version}, fh, indent=2)
    print(f"{prefix}: ready -> {python}")
    return python


def verify_runtime(python: str, prefix: str = "VENV") -> bool:
    """Confirm the pinned interpreter can actually IMPORT the stack and see the GPU.

    Checking version metadata is necessary but not sufficient, and the gap is exactly where this
    project's failures came from: `importlib.metadata.version("pyarrow")` answers "21.0.0" while
    `import datasets` raises AttributeError, and a pin table can read all-OK while `import torch`
    dies on a missing shared library. So import for real, in a child, with the patched library path.

    Returns True when torch imports, CUDA state is reported, and transformers/datasets/peft import.
    """
    env = patch_env(prefix=prefix)
    code = (
        "import json;"
        "import torch, transformers, datasets, peft, numpy, pyarrow;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'transformers': transformers.__version__,"
        "'datasets': datasets.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env)
    except OSError as e:
        print(f"{prefix}: could not launch {python}: {e}")
        return False
    if r.returncode != 0:
        print(f"{prefix}: the pinned interpreter CANNOT import the stack. Last error:")
        print("\n".join(r.stderr.strip().splitlines()[-12:]))
        print(f"{prefix}: this is usually a missing shared library on the loader path. On a conda")
        print(f"{prefix}: image the fix is the library path patch_env() applies -- if it still fails,")
        print(f"{prefix}: the base image needs the conda libs installed.")
        return False
    got = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"{prefix}: imports OK -- torch {got['torch']}, transformers {got['transformers']}, "
          f"datasets {got['datasets']}")
    print(f"{prefix}: cuda available = {got['cuda']}" + (f"  ({got['gpu']})" if got["gpu"] else ""))
    return True


def _read_pins(requirements: str):
    """[(name, version)] straight from the requirements file -- the same parser environment.py uses,
    so there is no second copy of the pins that can drift."""
    pins = []
    with open(requirements, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            name, sep, version = line.partition("==")
            if sep:
                pins.append((name.strip(), version.strip()))
    return pins


def describe(python: str, requirements: str = REQUIREMENTS, prefix: str = "VENV") -> bool:
    """Print the venv's resolved versions. Returns True when every pin is satisfied.

    Runs in a SUBPROCESS on purpose.  Importing the pinned packages into THIS process to read their
    versions is exactly the failure mode we are avoiding -- a stale numpy or pyarrow already bound in
    the kernel cannot be fixed by importing a second copy.
    """
    pins = _read_pins(requirements)
    # The child reads the same file, so the two can never disagree about what "satisfied" means.
    code = (
        "import importlib.metadata as m, json, sys;"
        f"pins=[tuple(l.split('==')) for l in open({requirements!r})"
        ".read().splitlines() if l and not l.startswith('#') and '==' in l];"
        "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])),"
        "'packages': {n.strip(): m.version(n.strip()) for n, _ in pins}}))"
    )
    if not os.path.exists(python):
        print(f"{prefix}: no interpreter at {python} -- the venv does not exist yet. "
              f"Run ensure() (or drop --check_only) to build it.")
        return False
    try:
        out = subprocess.check_output([python, "-c", code], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"{prefix}: {python} could not report its versions:\n{e.output}")
        return False
    got = json.loads(out)
    print("=" * 78)
    print(f"{prefix}: resolved package versions in the LOCAL SUBPROCESS interpreter")
    print("=" * 78)
    py_ok = tuple(map(int, got["python"].split("."))) >= (3, 11)
    print(f"python      {got['python']:12s} {'OK' if py_ok else 'TOO OLD (need >= 3.11)'}")
    ok = py_ok
    for name, want in pins:
        have = got["packages"].get(name)
        good = have == want
        ok &= good
        mark = "OK" if good else ("MISSING" if have is None else "MISMATCH")
        print(f"{name:15s} want {want:10s} have {str(have):12s} {mark}")
    return ok


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--root", default=os.path.join(os.getcwd(), "decider_v2_work", "venv"))
    ap.add_argument("--requirements", default=REQUIREMENTS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args(argv)

    if args.check_only:
        py = _venv_python(args.root)
        ok = describe(py, args.requirements) and verify_runtime(py)
        return 0 if ok else 1
    py = ensure(args.root, args.requirements, force=args.force)
    ok = describe(py, args.requirements) and verify_runtime(py)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())