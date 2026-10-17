<br/>
<p align="center">
  <h1 align="center">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</h1>
  <p align="center">
    CVPR 2025
  </p>
  <p align="center">
    <a href="https://yyuncong.github.io/">Yuncong Yang</a>,
    <a href="https://hanyangclarence.github.io/">Han Yang</a>,
    <a href="https://www.linkedin.com/in/jiachen-zhou5/">Jiachen Zhou</a>,
    <a href="https://peihaochen.github.io/">Peihao Chen</a>,
    <a href="https://icefoxzhx.github.io/">Hongxin Zhang</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2411.17735">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://umass-embodied-agi.github.io/3D-Mem/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

---

This is the official repository of **3D-Mem**: 3D Scene Memory for Embodied Exploration and Reasoning.

![](assets/teaser.png)

---

## News

- [2025/03] Inference code for A-EQA and GOAT-Bench is released.
- [2025/02] 3D-Mem is accepted to CVPR 2025!
- [2024/12] [Paper](https://www.arxiv.org/abs/2411.17735) is on arXiv.

## Installation
Set up the conda environment (Linux, Python 3.9):
```bash
conda create -n 3dmem python=3.9 -y && conda activate 3dmem

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2 python-dotenv

```


## Run Evaluation

### 1 - Preparations

#### Dataset
Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify
the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that 
contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, you can set the `scene_data_path` in the config files as `/your_path/hm3d/`.

The test questions of A-EQA and GOAT-Bench are provided in the `data/` folder. For A-EQA, we provide two subsets of different size: `aeqa_questions-41.json` and `aeqa_questions-184.json`, where `aeqa_questions-184.json` is the official subset provided by OpenEQA and `aeqa_questions-41.json` is a smaller subset for quick evaluation.
For GOAT-Bench, we include the complete `val_unseen` split in this repository.

#### Gemini API Setup
The evaluation helpers use the OpenAI Python client with Gemini's OpenAI-compatible endpoint,
`https://generativelanguage.googleapis.com/v1beta/openai/`.

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey).
Prashansa, edit the repository-root `.env` file locally and fill in `GEMINI_API_KEY`;
the supplied file deliberately has an empty key. If `.env` is absent, create it using
[`.env.example`](.env.example), which contains only these placeholders:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_MIN_REQUEST_INTERVAL=15
GEMINI_MAX_REQUEST_ATTEMPTS=3
GEMINI_MAX_REQUESTS_PER_RUN=20
GEMINI_MAX_OUTPUT_TOKENS=512
```

`python-dotenv` (included in the installation command above) automatically loads this
repository-root `.env`, regardless of the current working directory. Existing environment
variables take precedence over `.env` values. `GEMINI_MODEL` selects the model and defaults
to `gemini-3.1-flash-lite` when unset. A missing or blank key raises a clear error before an API
client is constructed. No source-code credential changes are needed; `OPENAI_KEY` remains
an alias for the Gemini key for existing callers.

Keep `.env` private: it is ignored by Git, and `.env.example` must never contain a real key.
Evaluation transmits prompts and scene images (including base64-encoded views, snapshots,
and goal images when used) to Google. Confirm that you may share these inputs, review the
provider's data policies and pricing, and check your quota before running: requests and
retries can incur API costs. This setup does not make any API calls on its own.

The default [Gemini 3.1 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
is a stable image-input/text-output model with a free tier and low paid prices; see
[Google's pricing](https://ai.google.dev/gemini-api/docs/pricing). There is no automatic
model or paid-tier fallback. Confirm model access and billing settings in your project.

Both evaluation helpers share one request guard. It enforces a **fixed 15-second
minimum** between request starts; `GEMINI_MIN_REQUEST_INTERVAL` can make the interval
longer, not shorter. The smoke launcher enforces at least **30 seconds**. Defaults
outside the launcher allow at most 3 attempts per logical request and 512 output
tokens. SDK retries are disabled, and the client's network timeout is 60 seconds.
Only rate limits (429), server errors (5xx), and connection/timeouts are retried,
with backoff of at least 30, then 60 seconds, or longer server-specified delays.
Permanent API errors fail immediately. Exhausted failures block further requests in
that process, preventing outer evaluation retries from restarting the retry cycle.

There are **two separate send caps**, counting failed sends, retries and requests
following empty responses across both helpers:

- `GEMINI_MAX_REQUESTS_PER_RUN` is configurable and per process (default 20;
  smoke at most 10). Only this counter resets when the process restarts.
- A **fixed 20-attempt rolling-24-hour cap** persists across processes and restarts
  using ignored repository-root `.runtime/gemini_budget.json` and its lock file.
  Attempts are recorded before sending. This cap cannot be increased through `.env`.
  Shared file locking also serializes requests from the same checkout.

Explicit daily-quota exhaustion or zero-quota 429 responses immediately block the
process and persist a cooldown of **at least 24 hours**. Other exhausted temporary
failures persist a cooldown of **at least 60 seconds**; server-specified delays may
extend either cooldown. Restarting does not clear the persistent cap or cooldowns,
and a rolling 24 hours is not Google's calendar-day quota reset.

**Do not delete, edit, reset or replace `.runtime/` or its ledger/lock to retry.**
Unreadable, invalid or unsafe state fails closed (no request is sent). A clock moving
backwards, including a monotonic-clock reset after reboot, can also fail closed.
Prashansa, stop and seek help inspecting the cause rather than resetting the ledger
or repeatedly restarting evaluation. `--check` does not validate or clear this ledger.

The configurable interval must be finite and positive; attempts, process cap and
output-token limit must be positive integers. Invalid settings fail before any
request. Short output limits can truncate answers. Run only one evaluation at a
time to keep GPU/memory use modest. These guards cover this checkout, not other
checkouts, machines or clients using your project. [Google's rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
also include per-project input-token quotas. Image-heavy requests may exceed them
even with pacing. These settings do not guarantee free-tier access or prevent all
charges; check your actual project quotas before changing limits or retrying.

### Safe one-question minival smoke test

Prashansa, activate the existing `3dmem` environment, then use:

```bash
conda activate /data/miniconda3/envs/3dmem
python run_smoke_test.py --help
python run_smoke_test.py --check
python run_smoke_test.py --run
```

YOLO-World also requires the official OpenAI `clip` package (not a similarly named
PyPI package). After inspecting its setup code, install the pinned source into the
active environment if missing:

```bash
python -m pip install --no-deps 'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
```

This installs only that reviewed revision, without upgrading its dependencies;
the existing environment must already supply its dependencies. The launcher sets
`YOLO_AUTOINSTALL=false` so Ultralytics cannot silently install missing packages.

`--help` explains the options. `--check` checks required packages, scene files,
configuration and API-key presence **without API calls, downloads, model loading,
or printing credentials**. No arguments also prints help. It uses the Python you
invoked, not a hardcoded environment. From another directory, use the launcher's
absolute path; it resolves repository paths itself. Checks do not test binary/GPU
compatibility, provider access, quotas, or the correctness of model predictions.

Only `--run` launches evaluation: the first of 184 questions sorted by question ID,
scene `00802-wcojb4TFT35`, for **at most three navigation steps**, with visualization
disabled. Camera resolution stays at the original 1280×1280; planner geometry is
unchanged. The smoke config opts into `sam_autocast: true`, enabling CUDA mixed
precision only around SAM prediction to reduce peak memory on the 8 GB GPU.
It still uses the same SAM-L weights; baseline configs are untouched and the
evaluator's default is false. Lower precision may change segmentation outputs, so
this is a smoke-only compatibility setting, not a baseline-equivalent result.
The initial offline perception probe ran out of CUDA memory at step 0 in SAM-L's
FP32 global attention, before Gemini or ledger access. With autocast, the offline
probe passed and the live three-step pipeline subsequently completed on this GPU.
The separate `cfg/eval_aeqa_smoke.yaml` leaves your minival config alone.
The launcher requires all four scene files and the minival scene-dataset config;
the existing `data/hm3d/val` symlink must resolve to the extracted minival scenes.

Every run gets a fresh `results/smoke/exp_eval_aeqa_smoke_<timestamp>_<id>/`
directory and saved launch config, avoiding resumed/skipped questions and overwritten
logs. Exported subprocess limits enforce at least 30 seconds between API sends,
one attempt per request, at most 10 sends per run and 512 output tokens; stricter
environment/`.env` settings win. The configured model (default
`gemini-3.1-flash-lite`) is retained, with no automatic model or paid-tier fallback.
These bounds are not a dollar or input-token budget. The shared guard's fixed
15-second floor, persistent 20-attempt rolling-24-hour cap and persistent cooldowns
still apply; never reset `.runtime/` to make another smoke run. Its network timeout
is 60 seconds. A 900-second launcher timeout or Ctrl+C stops the evaluation
process group; a request already received by Google cannot be undone.

Missing model weights are warnings during `--check`; `--run` may download the
official Ultralytics YOLO/SAM, LAION OpenCLIP and OpenAI CLIP artifacts through the
installed libraries. Only proceed if you trust these upstream weight sources and
their licenses: model checkpoints can contain executable serialized objects. The
launcher installs nothing and has no separate `--prepare-models` mode. Explicit
HF/Torch/YOLO/XDG/Matplotlib caches/configs use ignored repository `.cache/`.
`HOME` is preserved: OpenAI CLIP is the exception and reuses/downloads its weights
at `~/.cache/clip/ViT-B-32.pt`. Do not assume all caches are project-local. Thread pools are
capped at two. An 8 GB GPU can still run out of memory; no successful GPU run is
implied by preflight. `--run` sends scene images/prompts to Google and may incur
charges. Review fresh logs before retrying, particularly after quota failures.

**2026-10-17 result:** the pipeline exited 0 after three steps and six Gemini
HTTP 200 responses, with no retries or observed 429 errors. The agent moved, but
did **not** solve the question within the short step cap (0/1 task success).
Pipeline completion is not answer accuracy or a benchmark score. See
[the completed smoke-test report](docs/SMOKE_TEST_REPORT.md) for timings, local
evidence links, safety limits, the earlier OOM and remaining risks.

### 2 - Run Evaluation on A-EQA

First run the following script to generate the predictions for the A-EQA dataset:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
To split tasks, you can add `--start_ratio` and `--end_ratio` to specify the range of tasks to evaluate. For example,
to evaluate the first half of the dataset, you can run:
```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```
After the scripts finish, the results from all splits will be automatically aggregated and saved.

To evaluate the predictions with the pipeline from OpenEQA, you can refer to [link](https://github.com/yyuncong/3D-Mem-AEQA-Eval)

### 3 - Run Evaluation on GOAT-Bench
You can directly run the following script:
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```
The results will be saved and printed after the script finishes. You can also split the task similarly by adding `--start_ratio` and `--end_ratio`.
Note that GOAT-Bench provides 10 explore episodes for each scene, and by default we only test the first episode due to the time and resource constraints.
You can also specify the episode to evaluate for each scene by setting `--split`.

### 4 - Save Visualization
The default evaluation config will save visualization results including topdown maps, egocentric views, memory snapshots, and frontier snapshots at each step. Although saving visualization is very helpful, it may slows down the evaluation process. Please make save_visualization false if you would like to run large-scale evaluation.


## Acknowledgement

The codebase is built upon [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs).
We thank the authors for their great work.

## Citing 3D-Mem

```tex
@InProceedings{Yang_2025_CVPR,
    author    = {Yang, Yuncong and Yang, Han and Zhou, Jiachen and Chen, Peihao and Zhang, Hongxin and Du, Yilun and Gan, Chuang},
    title     = {3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17294-17303}
}
```
