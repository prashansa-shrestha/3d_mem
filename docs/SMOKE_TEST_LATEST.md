# Latest smoke verification — 2026-10-18

Command: `/data/miniconda3/envs/3dmem/bin/python -B run_smoke_test.py --run`

The one-question, three-step run completed with exit code 0 in about 4 minutes
19 seconds. YOLO, SAM, CLIP, scene updates, navigation, Gemini calls and result
saving completed. Six API sends returned HTTP 200; one returned HTTP 503 and
succeeded after the existing 30-second retry backoff.

Results: `results/smoke/exp_eval_aeqa_smoke_20261018T035009_6732c76f/`
Log: `log_0.00_0.01.log` inside that directory.

Changes: the smoke launcher now allows up to three attempts per call and defaults
`PYTORCH_CUDA_ALLOC_CONF` to `max_split_size_mb:128`. Existing stricter request
settings and existing allocator settings take priority. The earlier live run
`exp_eval_aeqa_smoke_20261018T034744_639ff9a5` ran out of memory at step 1;
the final run completed all three steps with the allocator setting.

The agent used all three steps exploring and did not answer the question. The
recorded task success is 0/1. This verifies pipeline execution, not answer accuracy
or full-dataset evaluation. NumPy warnings about the mean of an empty successful
path-length list were nonfatal. No changes were made to baseline evaluation configs.

Offline smoke preflight, allocator-default/override checks, syntax validation and
`git diff --check` passed. No credentials, caches or generated results are committed.
