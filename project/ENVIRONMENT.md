# Environment setup (Windows, Python 3.10)

These are the exact steps and version pins that produce a working environment on
Windows. The order matters because of two legacy-dependency quirks (see notes).

```powershell
# from the repo root
py -3.10 -m venv venv

# build tools (setuptools < 66 is required to build gym 0.21)
venv\Scripts\python.exe -m pip install "setuptools==65.5.0" "wheel==0.38.4" "Cython==3.2.3" "numpy==1.23.5"

# parallelworkloads (Lublin99 / Tsafrir05 generators) — needs the rand48 POSIX
# compatibility shim to compile under MSVC. Install from the patched source.
venv\Scripts\python.exe -m pip install --no-build-isolation <path-to>\parallelworkloads

# this package (editable) + test/base deps
venv\Scripts\python.exe -m pip install --no-build-isolation -e . "intervaltree==3.0.2" "pytest==7.4.4" "pytest-cov==4.1.0" "coverage==7.14.1"

# gym 0.21 has invalid metadata that modern pip rejects -> use pip < 24.1
venv\Scripts\python.exe -m pip install "pip==23.3.2"
venv\Scripts\python.exe -m pip install --no-build-isolation "gym==0.21.0"

# deep RL training (CPU build)
venv\Scripts\python.exe -m pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python.exe -m pip install "tensorboard==2.20.0"
```

## Notes / gotchas

- **gym 0.21 needs `pip < 24.1`.** Newer pip refuses to install it
  (`invalid metadata: ... opencv-python (>=3.)`). Pin pip to 23.3.2 first.
- **gym 0.21, not 0.26.** `pip install -e .` may pull gym 0.26, whose API differs
  (5-tuple `step`, tuple `reset`). The code targets the 0.21 API — keep 0.21.
- **`setuptools==65.5.0`** (i.e. < 66) is required to build gym 0.21.
- **`parallelworkloads`** does not build under MSVC without the rand48 POSIX shim
  (`rand48_compat.h`). It is a separate package; a clean fork + upstream PR for the
  shim is still pending.
- **`pytest==7.4.4`**, not the 4.6.3 pinned upstream — 4.6.3 crashes on Python 3.10
  (`required field "lineno" missing from alias`).
- The SWF test (`TestSwfGenerator`) needs `test/LANL-CM5-1994-4.1-cln.swf.gz`
  (a plain-text SWF trace despite the `.gz` name; the parser reads it as text).

A frozen snapshot of the working environment is in `requirements-lock.txt`.

## Verify

```powershell
venv\Scripts\python.exe -m pytest schedgym\test_schedgym.py -q --no-cov   # 98 passed
venv\Scripts\python.exe eval_baselines.py --episodes 3
```
