# Environment setup (modern stack: uv + gymnasium + Rust)

As of the merge with upstream/master (July 2026), the project uses the
modernized toolchain: Python 3.13, gymnasium (not gym 0.21), `pyproject.toml`
with [uv](https://docs.astral.sh/uv/) for dependency management, and a
maturin/PyO3 Rust extension (`schedgym._schedgym_rs`) that accelerates the
Heap / IntervalTree / ResourcePool hot paths (pure-Python fallbacks exist,
but the build produces the Rust extension).

## Prerequisites (one-time, per machine)

- **uv** — `winget install astral-sh.uv`
- **Rust toolchain** — `winget install Rustlang.Rustup` (stable-msvc)
- **MSVC Build Tools** — required both for the Rust linker and for building
  the Cython extensions of `parallelworkloads`
- Python 3.13 (uv will find it, or download one; pinned in `.python-version`)

## Setup

```powershell
# from the repo root
uv sync --extra test          # core + test deps, builds Rust ext + parallelworkloads
uv sync --extra test --group rl   # add torch + tensorboard for training
```

That's it. `uv sync` creates `.venv/`, builds the project (maturin/Rust) and
`parallelworkloads` from the vendored source, and installs everything pinned
by `uv.lock`.

## Notes / gotchas

- **parallelworkloads is vendored** under `vendor/parallelworkloads`. It is
  upstream's source plus `rand48_compat.h` — a POSIX rand48 shim (guarded by
  `#ifdef _WIN32/_MSC_VER`) without which the package does not compile under
  MSVC. `[tool.uv.sources]` in `pyproject.toml` points at it. An upstream PR
  for the shim is still pending; if it is ever merged, the source can be
  switched back to the upstream git URL.
- **gymnasium API**: `reset()` returns `(obs, info)` and takes `seed=`;
  `step()` returns `(obs, reward, terminated, truncated, info)`. All envs,
  tests and scripts in this repo were ported accordingly.
- **CompactRmEnv observation bounds**: empty job slots are padded with -1
  sentinels which the SMDP log transform maps to small negatives, and
  log-scaled event time offsets can slightly exceed 1.0 — the declared Box
  is therefore `low=-1.0, high=inf`. (This was always true numerically; the
  old gym stack simply never checked.)
- The SWF test (`TestSwfGenerator`) needs `test/LANL-CM5-1994-4.1-cln.swf.gz`
  (a plain-text SWF trace despite the `.gz` name; the parser reads it as
  text). It is committed in this repo.
- The legacy setup (Python 3.10 / gym 0.21 / pip pins) is documented in the
  git history of this file, and `project/requirements-lock.txt` remains the
  frozen snapshot of that environment, should the old runs ever need to be
  reproduced exactly.

## Verify

```powershell
uv run pytest schedgym -q --no-cov        # 181 passed
uv run python eval_baselines.py --episodes 3
uv run python -c "import schedgym._schedgym_rs"   # Rust backend present
```
