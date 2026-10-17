# Run 3D-Mem: small end-to-end tutorial

Prashansa, this uses your existing environment, downloaded minival scenes, model
weights, and Gemini configuration. It runs **one question for up to three steps**:
camera observations → object detection → scene memory → Gemini reasoning → navigation.
It is a functionality test, not the full benchmark.

## 1. Open the project and activate Python

```bash
cd /data/Prashansa/major/refs/3D-Mem
conda activate /data/miniconda3/envs/3dmem
```

`cd` selects the repository. Activation selects the Python environment containing
the installed dependencies. Use your normal GPU-enabled terminal and keep internet
access available. Your Gemini key should remain in the repository-root `.env`;
the code loads it automatically. Do not paste the key into commands or Git.

## 2. Check the setup without using API quota

```bash
python run_smoke_test.py --check
```

This checks dependencies, dataset paths, configuration, and key presence. It does
not send requests, download weights, or test the GPU. Fix any `ERROR` before
continuing; a missing-weight `WARNING` means the run may need an official download.

## 3. Run the real pipeline

```bash
python run_smoke_test.py --run
```

This loads the models and scene, builds memory, sends text/images to Gemini, and
moves the simulated agent. It uses `cfg/eval_aeqa_smoke.yaml`, including the
SAM mixed-precision setting needed for the laptop's 8 GB GPU. Baseline configs
are unchanged. **Real requests consume your available Gemini quota and may incur
charges depending on your existing account settings.** No paid fallback is enabled.

The verified run took about **3½ minutes** and made **six successful requests**.
Timing and model decisions can vary. Wait during request-spacing pauses; do not
launch another evaluation alongside it. Press **Ctrl+C** to stop the run.

## 4. Read the results

The launcher prints a fresh output directory under:

```text
results/smoke/exp_eval_aeqa_smoke_<timestamp>_<id>/
```

Open `log_0.00_0.01.log` there to see steps, requests, navigation, and the question
outcome. JSON files record answers and memory counts. `All scenes finish` means
execution finished, **not necessarily that the answer was correct**. The verified
three-step test finished normally but did not solve the question; its answer JSON
was empty. See [the run report](SMOKE_TEST_REPORT.md) for details.

## Safety and troubleshooting

- Smoke limits: at least **30 seconds** between requests, **one attempt** per call,
  at most **10 attempts per run**, **512 output tokens**, and a **15-minute timeout**.
- The shared guard also enforces **20 attempts per rolling 24 hours** across runs
  from this checkout. Failed requests count. Other clients' usage and Google's
  input-token quotas are not covered, so rate-limit avoidance is not guaranteed.
- On a quota error, guard error, or crash, stop and inspect the log rather than
  repeatedly restarting. **Never delete or edit `.runtime/` to reset the budget.**
  Clock changes or an interrupted request can make the guard stop safely.
- If activation is unavailable, use `/data/miniconda3/envs/3dmem/bin/python`
  instead of `python` in the commands above.
- For help without running anything: `python run_smoke_test.py --help`.
