# CUHK-X Small Model Track — Submission and Reproduction

Full Chinese instructions: [README_zh-CN.md](README_zh-CN.md).

This package preserves the frozen solution reported by the participant as **0.91542** on
Kaggle. This is not a promised score on unseen data. Read `docs/RELEASE_STATUS.md`
before uploading. Code, model, bilingual instructions and training-resource disclosure
are provided. The organizer's honor declaration still needs the actual participant's signature.

Rules update: current organizer replies explicitly allow AI coding assistants,
model-generated test pseudo-labels and small YOLO preprocessing. See
`docs/RULES_LIVE_AUDIT.md`. AI development permission is no longer an unresolved item.

Clean-environment acceptance (2026-09-11): a newly extracted package, a venv without
global/user site-packages, and an independent raw-data copy reproduced all405
historical predictions byte-for-byte through the PowerShell entry point (349seconds).
See `docs/ACCEPTANCE_20260911.md` for the recorded steps and test scope. Historical
development-environment timing (224seconds) is not the clean-install measurement.

## 1. Two different deliverables

- For the September 15 Kaggle deadline: `kaggle/submission.csv` is the unchanged,
  previously scored 405-row CSV. Its SHA256 is
  `a9796cde28c8a999623a1ee388047c97d2d1553fd9bf10a9ed30f8ca7768561c`.
- For organizer verification: run `inference.sh` on the newly supplied raw data.
  **Never submit that historical CSV for a replacement test set.** The runtime does
  not read `kaggle/`, historical test caches, teacher targets, or research outputs.

The official website lists a September 22 23:59 UTC package deadline, but also says
48 hours after shortlist notification. Follow the organizer's written notification
and prepare by September 15. See `docs/RULES_LIVE_AUDIT.md` for the conflict.

## 2. Package contents

```text
checkpoints/model.pth         one checkpoint: Student + YOLO pose weights
code/runtime.py              fresh-data orchestrator
code/legacy/                 frozen preprocessing/model source dependencies
code/training/project/       full local training/teacher source and config snapshot
inference.sh                 Bash entry point
inference.ps1                Windows entry point
requirements.txt             observed dependency versions
requirements-lock.txt        complete tested Windows dependency lock
verify_package.py            integrity and submission-material checks
docs/                        audit, provenance, release limitations
kaggle/submission.csv         historical scored CSV; not a runtime dependency
```

`honor_declaration.pdf` must be supplied and signed by the actual team using the
organizer's form. No signature or official form has been fabricated in this package.

## 3. Environment

After extracting the final archive, first run `python verify_package.py` to check
the package hashes and required technical files. For submission, additionally run
`python verify_package.py --for-submission` after adding the organizer's signed
declaration; a PDF header check does not authenticate a signature.

Observed development environment: Windows, Python 3.12.7, NVIDIA RTX 5070 Ti Laptop
(12 GB VRAM), driver 581.29, PyTorch 2.9.1+cu130, torchvision 0.24.1+cu130.
Use at least16GB RAM and20GB free space including the environment, input copy and
scratch files for405 clips. The scratch
estimate increases with the number of clips. GPU is strongly recommended; CPU
support is not a guarantee of meeting the two-hour verification limit.

From the extracted package directory, create a clean environment using Python3.12
(do not enable system-site-packages), then install. Windows PowerShell:

```powershell
python --version
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

If script activation is restricted, use `.\.venv\Scripts\python.exe` instead of
`python` in the commands below; do not change the machine's global execution policy.
Linux uses `python3.12 -m venv .venv` followed by `source .venv/bin/activate`.
The interpreter version must be3.12. Then:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -r requirements-lock.txt
python -m pip check
python -c "import sys,site,torch,torchvision; print(sys.executable); print(site.ENABLE_USER_SITE); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

Confirm the printed Python path is inside `.venv`, user-site is `False`, and CUDA
is `True` for the GPU execution route. Installation needs internet access; inference needs no LLM, API key, account, model
download, or assistance from the developer. These commands describe the observed
environment, not a clean Linux installation test. For an offline judging machine,
prepare the corresponding wheels before the meeting. Do not install a different
Ultralytics version without rerunning the raw-data regression.

The original development environment had stale torchvision metadata. The new venv
does not reproduce that mixed installation: imported torch/torchvision versions and
pip dependencies were checked successfully. No original global packages were changed.

For installation errors such as `SSL: UNEXPECTED_EOF_WHILE_READING`, check the
configured proxy; do not disable TLS verification. Our inherited local proxy failed
while direct HTTPS access to the official index worked. Only on a network that
permits direct access, clear proxy variables in the current PowerShell session and
retry the same installation commands:

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
$env:NO_PROXY = '*'
```

NO_PROXY prevents automatic fallback to the Windows system proxy. This changes only
the current process environment, not global proxy settings.

## 4. Input contract

Extract the organizer's data. `DATA_DIR` may contain sample directories directly or
their parent directory named in the official CSV. Each sample directory contains
the original modality folders, e.g. `IR/`, `Skeleton/`, `IMU/`, `Depth_Color/`.
Do not rename the raw modality files: their intra-clip frame IDs/times align sensors.
They are not looked up against any external label mapping.

Supply the organizer's unfilled CSV with a `path` column and optionally an empty
`prediction` column. Preserve every `path` exactly. There is no fixed 405-row
assumption in the packaged entry point. No labels, predictions, hand corrections,
old manifests, or old feature caches may be supplied as input.

This frozen preprocessing requires a nonempty Skeleton recording with a recoverable
time axis. Missing/unreadable IR is masked; missing IMU is masked. The historical ROI
builder requires usable synchronized Depth/IR/Skeleton. If that combination is absent,
the visual branch is masked rather than secretly calling another model. Counter-only
Skeleton without an absolute timestamp source is rejected explicitly.

```text
new_data/
  test.csv
  small_model_track_test/
    SM_test_0001/
      IR/
      Skeleton/
      IMU/
      Depth_Color/
    ...
```

If the organizer changes modality naming/schema or does not provide a path list,
obtain their format instructions; do not guess labels or silently substitute old data.

## 5. Run — no interaction with an AI assistant

Linux / Bash, from any working directory:

```bash
bash /path/to/package/inference.sh /path/to/new_data /path/to/new_data/test.csv /path/to/new_output
```

Windows PowerShell:

```powershell
& .\inference.ps1 -DataDir 'D:\new_data' -TestCsv 'D:\new_data\test.csv' -OutputDir 'D:\new_output'
```

Both entry scripts prefer the package's `.venv` interpreter when it exists, even
without activation. PowerShell also accepts `-PythonExecutable` for an explicit
environment. If local `.venv` is absent, the active PATH Python is used; check it
before inference. Each output report records the actual interpreter path.

Equivalent explicit Python invocation:

```bash
python code/runtime.py --bundle checkpoints/model.pth --data-root /path/to/new_data --test-csv /path/to/new_data/test.csv --output /path/to/new_output
```

Submit `new_output/submission.csv`. It has exactly `path,prediction`, one integer
class in 0–39 per requested sample, in the input order. Wait for a successful exit;
do not upload a partial result after an error. Use a new output directory per run.
Keep the runtime log and validation reports with the result. `--keep-work` retains
newly generated intermediate caches for debugging; these contain competition data
and must not be publicly redistributed.

Check `new_output/report.json`: `partial` must be `false`, and `rows` must match the
organizer's CSV. `--max-rows` is a developer smoke-test option only; a partial run
must never be submitted. The report records input/model/output hashes and runtime.

## 6. Frozen computation and model budget

The pipeline is raw input → automatic YOLO pose → deterministic ROI → IR pixels
and Skeleton/IMU windows → compact MC3/MoBind Student → argmax CSV. There is no
inference-time teacher, session decoder, self-training, internet call or per-sample
manual decision. Newly issued data is predicted with the fixed checkpoint.

All inference weights, including pose preprocessing, are in one checkpoint. The
original Student-only file was 94,393,969 bytes; including separate pose weights
would exceed the strict decimal 100 MB threshold. This package stores Student
floating tensors in FP16 and includes pose weights in the same file, then loads
Student tensors into FP32 modules. `build_manifest.json` records the measured total
bytes and hash. This is an explicit quantized deployment variant, not a claim that
weight values are bit-identical to the historical Student.

## 7. Training history and scientific limitations

The Student was fitted on official training subjects, then adapted to automatically
generated pseudo-labels on the old unlabeled Kaggle inputs. Training-only teachers
and recording/session/repeat methods contributed to these targets. They are **not**
run on the newly issued test set and are not hidden inference dependencies.

`code/training/project/` contains the complete local source/config snapshot, including
teacher feature producers and training scripts; `code/training/source_manifest.json`
records hashes and static checks. Read `docs/TRAINING_PIPELINE.md` for the functional
training sequence and recorded adaptation command, and `docs/EXTERNAL_RESOURCES.md`
for exact model IDs, downloads and licensing notes. `docs/provenance/` preserves
historical run summaries. Their paths are training records, not runtime configuration.
No new training experiment was run for this packaging task. This scope limitation is
not presented as an additional organizer requirement. Historical validation limitations
remain disclosed in `docs/LEAKAGE_AUDIT.md`.

Historical 2211/2470 validation performance must not be advertised as an independently
verified, fully nested end-to-end score. Upstream global-OOF and calibration reuse
limit that interpretation. No test ground truth was accessed for the package tests;
agreement with an old prediction CSV is a software regression, not accuracy evidence.

## 8. Before upload

Review the release status, obtain the outstanding organizer confirmations, include
the genuinely signed declaration, and verify package integrity. Do not claim
"certified compliant" solely from the byte-size check or a successful forward pass.
Do not publish original data, scratch caches, private dataset access credentials,
or training assets without their applicable permissions.

Optional model-free input/safety tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

18 tests cover input shape, all CSV rows, duplicate headers, paths and Unicode image
reading. Input validation and library-cache isolation were hardened for this release;
the trained weights and class predictions were not changed.
