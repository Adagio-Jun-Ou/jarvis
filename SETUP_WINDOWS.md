# Running Jarvis on Windows

The project was developed on Linux (`~/jarvis`). This documents the Windows setup.

## Layout

```
<repo>\
├── venv\                  # virtualenv — gitignored
└── .cache\                # all caches — gitignored
    ├── pip\               # pip download cache
    ├── hf\                # HuggingFace models (~1 GB after first test run)
    ├── torch\             # torch hub cache
    └── tmp\               # build temp
```

## One-time setup

Run these from the repository root:

```powershell
# 1. Create the venv
& "C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe" -m venv .\venv

# 2. Point every cache at the repo's .cache directory
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"
$env:HF_HOME       = "$PWD\.cache\hf"
$env:TORCH_HOME    = "$PWD\.cache\torch"
$env:TMP           = "$PWD\.cache\tmp"
$env:TEMP          = "$PWD\.cache\tmp"

$py = ".\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel

# 3. Runtime dependencies
& $py -m pip install pydantic pyyaml rich numpy httpx cryptography bcrypt `
    fastapi uvicorn websockets pillow beautifulsoup4 pandas matplotlib `
    scikit-learn psutil aiofiles python-multipart duckduckgo-search

# 4. PyTorch — CPU build only. run.py disables every GPU backend explicitly
#    (CUDA/HIP/ROCm/OpenCL), so the CUDA wheels would be ~2.3GB of dead weight.
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 5. Embeddings (unblocks the 46 tests that used to skip)
& $py -m pip install sentence-transformers transformers

# 6. Test + dev tooling
& $py -m pip install pytest pytest-asyncio pytest-timeout pytest-xdist pytest-cov ruff

# 7. Optional: music subsystem
& $py -m pip install pretty_midi music21 mido
```

Optionally persist the cache variables so later sessions keep using them:

```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "$PWD\.cache\hf", "User")
[Environment]::SetEnvironmentVariable("TORCH_HOME", "$PWD\.cache\torch", "User")
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "$PWD\.cache\pip", "User")
```

`run.py` also honours `JARVIS_CACHE_DIR` for the torch-inductor cache; point it at
a roomy volume if the system temp directory is constrained.

## Running

```powershell
$py = ".\venv\Scripts\python.exe"

& $py run.py --interface cli                          # terminal chat
& $py run.py --interface web --port 8099 --insecure   # web UI over HTTP
& $py run.py --interface web --port 8443 --https      # web UI over TLS (needs ssl/)
```

First web start takes ~2 minutes: the prefrontal cortex loads a ~100M-parameter
LCM and every background subsystem is started before the port opens. Wait for
`/health` rather than assuming failure:

```powershell
curl.exe -s http://127.0.0.1:8099/health   # {"status":"healthy","service":"jarvis"}
```

The Makefile targets assume POSIX (`./venv/bin/python`); on Windows use
`venv\Scripts\python.exe` directly as above.

## Tests

```powershell
$py = ".\venv\Scripts\python.exe"
& $py -m pytest tests\ -q -n 2 -m "not requires_host_env"
```

Two notes on test runs:

- **Parallelism is memory-bound, not CPU-bound.** With torch installed each
  xdist worker resident set is ~1GB. High `-n` values can exhaust memory;
  `-n 2` to `-n 4` is the safe range. Higher values show up as
  `node down: Not properly terminated`, which is an OOM, not a test failure.
- **First run downloads ~1GB of models** (`facebook/hubert-base-ls960`,
  `jinaai/jina-embeddings-v5-text-nano`) into `.cache\hf`. Subsequent runs are
  offline; set `HF_HUB_OFFLINE=1` to enforce that.

`requires_host_env` marks tests that assert facts about the host machine
(installed `gh`/`cargo`/`rustc`, a MuseScore soundfont, `~/.jarvis/secrets/`)
rather than Jarvis behaviour. They are excluded in CI and will fail locally
unless you have that tooling — that is expected.

## Windows-specific behaviour worth knowing

- **Console encoding.** `run.py` forces UTF-8 on stdout/stderr at startup.
  Without it the banner emoji raises `UnicodeEncodeError` under the default
  cp1252 codepage and kills the process *after* every subsystem has started.
- **File permissions.** `atomic_write_json` and the SSL key chmod to `0600`,
  which is a no-op on Windows — NTFS uses ACLs. Secrets written by Jarvis are
  therefore not permission-protected here the way they are on Linux. The two
  tests asserting `0600` are skipped on `nt` for this reason.
- **`core/sandbox.py`** degrades to timeout + output caps on Windows: the
  rlimit-based CPU/memory/process isolation is POSIX-only.
