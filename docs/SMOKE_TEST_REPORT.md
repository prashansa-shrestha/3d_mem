# A-EQA minival smoke test

Prashansa, the live smoke pipeline completed successfully on **2026-10-17**,
but **the question was not solved within the three-step limit**. This is a working
integration result, not a successful answer or benchmark-accuracy result.

## Final result

The primary task supplied the following completed-run evidence. Finalization of
this document did not rerun evaluation, send API requests or inspect credentials.

- Start: **2026-10-17 22:22:37 Nepal time (UTC+05:45)**, corresponding to
  `20261017T163737` UTC in the output-directory name.
- Process exit: **0**. Evaluator elapsed time was approximately **3 minutes
  30 seconds**, approximately **3 minutes 35 seconds** including imports/startup.
- Scope: one question, `025257b6-8b7e-4f6f-aacc-1788069cbfad`, scene
  `00802-wcojb4TFT35`; all three navigation steps **0, 1 and 2** ran.
- Gemini: **6 HTTP 200 responses** from `gemini-3.1-flash-lite`, **0 retries** and
  **0 observed HTTP 429 responses**. Rounded send times were 20, 63, 96, 128,
  161 and 192 elapsed seconds; gaps were 43, 33, 32, 33 and 31 seconds. These
  observations meet the smoke's 30-second request-start minimum; processing and
  response time also contribute to the gaps.
- Perception/navigation: all model weights and the renderer loaded. SAM autocast
  resolved the observed 8 GB FP32 OOM. Gemini selected valid frontiers and the
  agent moved on all three steps. Final memory contained **7 objects, 4 snapshots**;
  recorded total frames **9**, filtered snapshots **1**; travelled distance
  **3.014740958491813 m**.
- Task outcome: **0/1 successful questions** at the short step cap. The dataset's
  target answer is a **bin with a yellow lid**, but the agent was still searching
  and produced no final correct bin answer. `gpt_answer.json` is empty because
  the question was not completed. **No OpenEQA scoring was run.** Exit 0 means
  the pipeline finished normally, not that it answered correctly.
- Primary-reported final safety state: **6 recorded attempts**, `inflight=false`,
  cooldown **0**. Relative to the fixed local cap of 20, this left **14 local
  attempts** at that moment, ignoring other usage and subsequent runs. This is
  **not Google's remaining project quota**, a free-tier guarantee or permission
  to reset the ledger.

## Evidence and commands

The completed run used this exact command from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 /data/miniconda3/envs/3dmem/bin/python -u run_smoke_test.py --run
```

It uses the existing environment, avoids bytecode writes, disables the Hugging
Face Xet transfer backend and prints evaluator output without Python buffering.
`--run` explicitly permits the downloads/API operations needed for evaluation.
GPU execution required the normal GPU-enabled terminal, outside the restricted
executor; no system packages, `sudo`, driver changes, paid-plan/billing changes
or provider fallback were needed.

For ordinary use, activate the existing environment, check first, and only then
opt into a new run:

```bash
conda activate /data/miniconda3/envs/3dmem
python run_smoke_test.py --check
python run_smoke_test.py --run
```

Activation selects the installed Python/dependencies. `--check` inspects local
requirements without API calls or downloads; `--run` starts a new bounded smoke
and can incur charges. No arguments prints help and never runs evaluation.
The launcher's absolute path works from another directory.

Saved evidence is under
[`results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/`](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/):

- [Evaluator log](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/log_0.00_0.01.log).
- [Saved run YAML](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/exp_eval_aeqa_smoke_20261017T163737_371d4c0a.yaml).
- [Answer JSON (empty)](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/gpt_answer.json),
  [frame counts](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/n_total_frames.json),
  [snapshot counts](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/n_total_snapshots.json),
  and [filtered-snapshot counts](../results/smoke/exp_eval_aeqa_smoke_20261017T163737_371d4c0a/n_filtered_snapshots.json).
- Aggregated JSON and pickle files, plus split-specific outputs, remain in that
  directory. Evidence links are local and may not resolve in a remote checkout
  because results are ignored. No prompts, images or credentials are embedded here.

Warnings included a `timm` deprecation and an applied CuDNN workaround. Empty-mean/
`nanmean` path-metric warnings followed from having zero successful questions;
they were not an exception or a nonzero process exit.

## Setup

- Launcher: `run_smoke_test.py`; only `--run` starts evaluation. No arguments shows help.
- Interpreter: the active Python used to invoke the launcher; the existing environment
  is available at `/data/miniconda3/envs/3dmem` (not hardcoded by the launcher).
- Configuration: `cfg/eval_aeqa_smoke.yaml`, independent of the original untracked
  `cfg/eval_aeqa_minival.yaml`; three navigation steps maximum, visualization off,
  original 1280×1280 camera and unchanged planner/scene-graph settings.
- Smoke-only compatibility opt-in: `sam_autocast: true` requests CUDA mixed
  precision only for SAM prediction, using the same SAM-L weights. The evaluator
  default remains false and baseline configs are untouched. Lower precision can
  change segmentation outputs; this is not baseline-equivalent accuracy evidence.
- Selection: `--start_ratio 0 --end_ratio 0.006` over 184 sorted questions selects
  question `025257b6-8b7e-4f6f-aacc-1788069cbfad`, scene `00802-wcojb4TFT35`.
- Required assets: `.basis.glb`, `.basis.navmesh`, `.semantic.glb`, `.semantic.txt`
  under `data/hm3d/val/00802-wcojb4TFT35/`, plus
  `data/hm3d/hm3d_annotated_minival_basis.scene_dataset_config.json`.
- The local `data/hm3d/val -> .` symlink was present during inspection. Dataset
  contents and the original minival config are not modified by this implementation.

## Safety boundaries

`--check` inspects dependency availability without importing heavy packages. A
network-blocked lightweight child imports only settings/dotenv and OmegaConf to
validate the smoke config, key presence and configured limits. It never imports
the evaluator or constructs an API client; child diagnostics are suppressed to
avoid exposing credentials. No weights are loaded or downloaded. Missing weights
are warnings, while missing dependencies/assets/key or invalid config are errors.
Dependency discovery is not proof of binary compatibility or CUDA readiness.

`--run` exports a minimum 30-second request interval, one attempt, a maximum of
10 sends and 512 output tokens, retaining stricter local limits. It retains the
configured Gemini model, with no downgrade/paid fallback. The shared Gemini guard
has a fixed 15-second floor (the smoke launcher's 30-second minimum is stricter),
a 60-second network timeout, and a hard 20-attempt rolling-24-hour cap shared
across processes/restarts in this checkout. The smoke limit of 10 sends is a
separate per-process cap; restarting clears only that counter, not persistent state.
Failed sends and retries count, with attempts recorded before sending. These are
not input-token/cost caps.
Prompts/images are transmitted to Google only during explicitly enabled evaluation.

Ignored `.runtime/gemini_budget.json` and its lock file hold persistent safety
state. Explicit daily/zero-quota failures persist a cooldown of at least 24 hours;
other exhausted temporary failures persist at least 60 seconds, extended when the
server requests a longer delay. Do not delete, edit or reset the ledger, lock or
`.runtime/` directory to bypass a cap/cooldown. Unsafe/unreadable state, backwards
wall time or a monotonic-clock reset after reboot can fail closed. Prashansa, seek
help inspecting the cause instead of resetting state or repeatedly restarting.
Offline `--check` does not inspect, validate or clear the persistent ledger.

The evaluator runs in its own process group with a 900-second timeout. Timeout
and Ctrl+C terminate that group, escalating to a kill if necessary; requests already
accepted by Google cannot be recalled. Startup and cleanup are protected as one
lifecycle: Ctrl+C during child creation is deferred until the child handle is
captured, then the process group is terminated before returning. Each run saves a unique launch configuration
and separate evidence under ignored `results/smoke/`. Explicit HF/Torch/YOLO/XDG/
Matplotlib caches/configs use ignored `.cache/`, and CPU thread pools are capped at
two. `HOME` is unchanged; OpenAI CLIP reuses or downloads its weights at the default
`~/.cache/clip/ViT-B-32.pt`, so not all caches are project-local.
No automatic package installation or separate preparation mode is provided.
`YOLO_AUTOINSTALL=false` explicitly disables Ultralytics' automatic dependency
installation. The primary task reports installing the inspected official OpenAI
CLIP source with `--no-deps`, pinned to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
This was the added dependency for the smoke setup; the earlier NumPy fix was
separate work, not a change made during this report finalization.

On an explicitly requested run, the installed model libraries may fetch official
Ultralytics, LAION and OpenAI model artifacts. Trust upstream checkpoint sources
and licenses before running: loading serialized weights is a trust boundary.

## Verification scope

- Offline: help/no-argument behavior, `--check` exit status and outstanding dependencies.
- Offline: config parsing, one-question selection, all four scene files and symlink paths.
- Offline: syntax, bounded child environment, preserved stricter settings and clean whitespace.
- Simulated child: timeout/Ctrl+C cleanup, including a grandchild, with no API or downloads.
- Outside a GPU-restricted sandbox: CUDA availability and modest GPU-memory usage.
- Completed live run: official download sources, configured model, fresh output
  directory, step count, request count and failed question outcome recorded above.
- Primary-reported guard regressions passed: persistent cap, request spacing,
  corruption/fail-closed handling, restart persistence and related regressions.
- Independent fresh review returned **SHIP before the live run**. Local simulation
  evidence below is separate from that independent verdict.

## Launcher verification performed (offline)

Using `/data/miniconda3/envs/3dmem/bin/python`:

- `run_smoke_test.py --help` passed. `--check` passed, including invocation by
  absolute path from `/tmp`, without evaluation, model loading or downloads.
- No-argument mode was verified not to call preflight. A failed preflight was
  verified not to launch evaluation. Configured-model export and new-file
  whitespace checks passed, as did `git diff --check`.
- AST syntax parsing passed. OmegaConf parsing confirmed the smoke config differs
  from the original minival config only in experiment/output names, model cache
  paths, the three-step limit and the later smoke-only `sam_autocast` opt-in;
  camera/planner/scene-graph settings are unchanged. Launcher validation requires
  `sam_autocast: true`; saved configs retain it.
- Real offline settings probes confirmed baseline limits become 30 seconds,
  1 attempt, 10 sends and 512 output tokens, and stricter settings (60 seconds,
  4 sends and 128 tokens) remain stricter. Only key presence was reported, never
  a credential value. Selected model is exported unchanged to the evaluator.
- Verified `HOME` stays unchanged and `YOLO_AUTOINSTALL=false` is exported;
  installed Ultralytics source confirms that switch disables auto-installation.
- Temporary-root simulations replaced only the evaluator command with local
  sleeping child/grandchild processes: timeout returned 124, simulated Ctrl+C
  returned 130, and normal exit returned 0. Process-group cleanup left neither
  child nor grandchild running, including when the group leader exited first.
  Each simulation created a unique three-step launch config under its temporary
  results directory. No simulator ran the actual evaluator or made API requests.
- After a fresh review identified a startup-interruption gap, the launcher was
  fixed and a boundary regression passed: a real SIGINT was delivered after the
  local child/grandchild existed but before the mocked `Popen` returned its handle.
  The launcher returned 130, restored the previous SIGINT handler and left neither
  process running. Timeout, runtime SIGINT and normal-exit cleanup also passed
  again alongside environment/config checks. This is local simulation evidence,
  not a live evaluation or a fresh post-fix reviewer verdict.
- This verification created no tests directory, made no installs or commits, and
  did not alter datasets or the original minival config.

### Rerun the saved offline simulations

The original simulations were inline commands; their temporary roots were cleaned
up automatically, so there was no original saved `/tmp` test script. Equivalent
checks, plus the startup-boundary regression, are now saved at
`/tmp/prashansa_smoke_launcher_checks.py`. Run from any working directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /data/miniconda3/envs/3dmem/bin/python /tmp/prashansa_smoke_launcher_checks.py /data/Prashansa/major/refs/3D-Mem
```

This uses the existing environment, prevents bytecode writes, probes only local
settings/configuration and substitutes local sleeping processes for the evaluator.
It checks safe/stricter environment limits and kills only process groups created
by its simulations. Simulation directories are created under `/tmp` and removed
automatically. The script does not import the Gemini client, contact APIs, download
weights, install packages, or access/reset the real `.runtime` ledger. It requires
the existing minival config for the baseline comparison. `/tmp` is temporary; this
script may not survive system cleanup or reboot.

## Primary-task evidence supplied during implementation

The following evidence was reported by the primary task, not reproduced by the
launcher implementer:

- Outside the restricted sandbox: RTX 4060 Laptop GPU, 8188 MiB total and about
  7159 MiB free; PyTorch CUDA availability was true. CUDA false inside the sandbox
  was device isolation, not evidence of a broken driver.
- Habitat rendered a local 128×128 RGB/depth/semantic frame in scene
  `00802-wcojb4TFT35`, and navmesh preflight passed. RGB standard deviation was
  44.28. This local rendering made no API calls and is not the three-step smoke run.
- Official YOLO/SAM downloads target `.cache/models/yolov8x-world.pt` and
  `.cache/models/sam_l.pt`. The primary task subsequently confirmed both downloads
  complete and ZIP CRC checks passed: YOLO 147,961,954 bytes, SAM 1,249,524,607 bytes.
  Reported SHA-256 values:
  - YOLO: `9b99398e46cffbf2b9a7e668512fa295f0d710d173ae0a815ec706ced5d1099b`.
  - SAM: `3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622`.
  These identify the downloaded artifacts; CRC/hashes alone are not a guarantee
  that serialized weights are safe. Cache artifacts must never be committed.
- LAION OpenCLIP weights were downloaded under `.cache/huggingface/hub`, revision
  `1a25a446712ba5ee05982a381eed697ef9b435cf`.
- OpenAI CLIP ViT-B/32 is now reported cached at the default
  `~/.cache/clip`, outside the project. At Prashansa's request,
  the launcher preserves `HOME` and reuses that default cache, rather than forcing
  a duplicate download. Do not claim every setup cache is project-local.
- No Gemini API calls had been made during those initial offline checks; the
  completed live run above subsequently made six real requests.
- The primary task independently reports `--check` and `--help` passing. The
  persistent guard was reviewed before the live run.

### Initial offline perception failure and memory-compatibility fix

The primary task reports that the actual offline perception probe failed with
CUDA out-of-memory at navigation step 0, inside SAM-L FP32 global attention,
before any Gemini request or ledger access. The failure log is:

[Initial FP32 OOM log](../results/offline_compatibility/offline_compatibility_20261017T163409/log_0.00_0.01.log)

LAION OpenCLIP loaded from the cached revision recorded above. The failed probe
does not establish a successful three-step run. To isolate the memory fix, the
smoke config now enables only the opt-in SAM-prediction autocast setting, retaining
1280×1280 rendering, SAM-L weights and all planner geometry. Evaluator support is
separately owned in `src/scene_aeqa.py`, defaulting off outside an explicit opt-in.
The subsequent offline rerun is recorded below; the final live pipeline outcome
is recorded at the top of this report.

### Second offline perception probe: passed to the Gemini boundary

The primary task reports that the second offline probe, with `sam_autocast: true`,
passed model loading and scene initialization, processed all seven views at step 0,
and completed mapping with 4 objects and 3 snapshots. Actual GPU runtime was
approximately 20 seconds. Its log is:

[Autocast offline-pass log](../results/offline_compatibility/offline_compatibility_20261017T163602/log_0.00_0.01.log)

The probe stopped intentionally at the Gemini boundary, before the first API
request, with no ledger access. This supports offline perception compatibility
with SAM autocast; it is not a completed three-step run, a live Gemini success,
or benchmark accuracy evidence. The prior FP32 OOM remains recorded above.

Fresh reviewer Goodall subsequently returned **SHIP**, as reported by the primary
task, before actual API execution. The completed live result replaces the earlier
progress-only status and is recorded at the top of this report. Runtime code
remained frozen during this documentation finalization. No further API calls,
installs, reruns or commits were made to finalize the report. No OpenEQA score or
broader accuracy claim is available from this one-question smoke.

## Limits and remaining risks

This run demonstrates the pipeline operating on this machine and one scene, not
general reliability, baseline-equivalent segmentation or task accuracy. Autocast
changes precision only in the smoke opt-in; baseline configs remain unchanged.
An 8 GB GPU can still exhaust memory on different scenes or workloads.

The persistent ledger protects this checkout only. External clients, checkouts
and machines may consume the same provider project quota, which is unknown here.
Image input tokens can hit TPM limits despite request spacing. No free-tier,
zero-cost or accuracy guarantee follows from six HTTP 200 responses. Never reset
`.runtime/` to obtain more attempts. Reboot/clock rollback can fail closed; seek
help inspecting the cause rather than deleting state. Keep results, private data,
credentials and model caches out of commits.
