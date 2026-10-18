"""Offline preflight and explicitly opted-in, bounded A-EQA smoke evaluation."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "cfg/eval_aeqa_smoke.yaml"
SCENE = "00802-wcojb4TFT35"
DEPENDENCIES = (
    "dotenv", "omegaconf", "numpy", "torch", "torchvision", "open_clip",
    "ultralytics", "clip", "habitat_sim", "quaternion", "open3d", "faiss",
    "supervision", "cv2", "PIL", "scipy", "sklearn", "skimage", "HiPart",
    "statsmodels", "treelib", "matplotlib", "openai", "httpx", "pytorch3d",
)
PROBE = r'''
import json
import sys

def offline(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind",
                 "subprocess.Popen", "os.system"}:
        raise RuntimeError("Preflight forbids network and child execution")

sys.addaudithook(offline)
try:
    from omegaconf import OmegaConf
    from src import const
    cfg = OmegaConf.load("cfg/eval_aeqa_smoke.yaml")
    expected = {
        "num_step": 3, "save_visualization": False,
        "exp_name": "exp_eval_aeqa_smoke", "output_parent_dir": "results/smoke",
        "questions_list_path": "data/aeqa_questions-184.json",
        "scene_dataset_config_path": "data/hm3d/hm3d_annotated_minival_basis.scene_dataset_config.json",
        "scene_data_path": "data/hm3d",
        "concept_graph_config_path": "cfg/concept_graph_default.yaml",
        "yolo_model_name": ".cache/models/yolov8x-world.pt",
        "sam_model_name": ".cache/models/sam_l.pt",
        "sam_autocast": True,
    }
    if any(cfg.get(name) != value for name, value in expected.items()):
        raise ValueError("Smoke configuration changed")
    print(json.dumps({
        "key_present": bool(const.OPENAI_KEY.strip()),
        "model_present": bool(const.VLM_MODEL.strip()),
        "model": const.VLM_MODEL,
        "limits": {
            "GEMINI_MIN_REQUEST_INTERVAL": const.GEMINI_MIN_REQUEST_INTERVAL,
            "GEMINI_MAX_REQUEST_ATTEMPTS": const.GEMINI_MAX_REQUEST_ATTEMPTS,
            "GEMINI_MAX_REQUESTS_PER_RUN": const.GEMINI_MAX_REQUESTS_PER_RUN,
            "GEMINI_MAX_OUTPUT_TOKENS": const.GEMINI_MAX_OUTPUT_TOKENS,
        },
    }))
except Exception:
    print(json.dumps({"error": True}))
    sys.exit(1)
'''


def child_environment():
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    cache = ROOT / ".cache"
    for name, relative in {
        "XDG_CACHE_HOME": "", "XDG_CONFIG_HOME": "config",
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "HF_DATASETS_CACHE": "huggingface/datasets", "TORCH_HOME": "torch",
        "YOLO_CONFIG_DIR": "ultralytics", "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba", "TRANSFORMERS_CACHE": "huggingface/hub",
    }.items():
        environment[name] = str(cache / relative)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["DO_NOT_TRACK"] = "1"
    environment["YOLO_AUTOINSTALL"] = "false"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        try:
            threads = int(environment.get(name, "2"))
        except ValueError:
            threads = 2
        environment[name] = str(max(1, min(2, threads)))
    return environment


def preflight(environment):
    print(f"Prashansa, checking with Python {sys.version.split()[0]}: {sys.executable}", flush=True)
    errors = []
    if sys.version_info < (3, 9):
        errors.append("Use the documented Python 3.9 environment or newer.")
    if os.name != "posix":
        errors.append("This launcher requires Linux/POSIX process-group cleanup.")
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Missing dependencies in this Python: " + ", ".join(missing))
    paths = [
        CONFIG, ROOT / "run_aeqa_evaluation.py", ROOT / "cfg/concept_graph_default.yaml",
        ROOT / "data/hm3d/hm3d_annotated_minival_basis.scene_dataset_config.json",
    ]
    scene_dir = ROOT / "data/hm3d/val" / SCENE
    for suffix in ("basis.glb", "basis.navmesh", "semantic.glb", "semantic.txt"):
        paths.append(scene_dir / f"wcojb4TFT35.{suffix}")
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty required file: {path.relative_to(ROOT)}")
    try:
        questions = json.loads((ROOT / "data/aeqa_questions-184.json").read_text())
        selected = sorted(questions, key=lambda question: question["question_id"])[:int(0.006 * len(questions))]
        if len(questions) != 184 or len(selected) != 1 or selected[0]["episode_history"] != SCENE:
            raise ValueError
        print(f"Question selection: first of 184, scene {SCENE}; at most three steps.")
    except (OSError, ValueError, TypeError, KeyError):
        errors.append("Question list must select exactly the expected first minival question.")
    settings = None
    try:
        probe = subprocess.run(
            [sys.executable, "-c", PROBE], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=20, check=False,
        )
        settings = json.loads(probe.stdout)
        if probe.returncode or settings.get("error"):
            raise ValueError
        if not settings["key_present"]:
            errors.append("GEMINI_API_KEY_LIST and GEMINI_API_KEY are missing/blank; set one locally in .env or your environment.")
        if not settings["model_present"]:
            errors.append("GEMINI_MODEL is blank; select your intended model explicitly.")
        environment["GEMINI_MODEL"] = settings["model"]
        for name, boundary in {
            "GEMINI_MIN_REQUEST_INTERVAL": 30,
            "GEMINI_MAX_REQUEST_ATTEMPTS": 3,
            "GEMINI_MAX_REQUESTS_PER_RUN": 10,
            "GEMINI_MAX_OUTPUT_TOKENS": 512,
        }.items():
            value = settings["limits"][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError
            bounded = max(value, boundary) if name.endswith("INTERVAL") else min(value, boundary)
            environment[name] = str(bounded)
        print("API key: presence checked privately; no client constructed or request sent.")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        errors.append("Lightweight settings/config check failed; verify python-dotenv, OmegaConf, smoke YAML and numeric Gemini limits. Details suppressed to protect secrets.")
    for name in ("yolov8x-world.pt", "sam_l.pt"):
        if not (ROOT / ".cache/models" / name).is_file():
            print(f"WARNING: {name} is not cached; --run may download official model weights.")
    clip_cache = ROOT / ".cache/huggingface/hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"
    if not any(clip_cache.glob("snapshots/*/open_clip_pytorch_model.bin")):
        print("WARNING: LAION OpenCLIP weights not found in the smoke cache; --run may download them.")
    if not (Path.home() / ".cache/clip/ViT-B-32.pt").is_file():
        print("WARNING: YOLO-World's OpenAI CLIP text weights are not cached; --run may download them (not an API call).")
    print("Preflight does not import the evaluator, load weights, test CUDA, install packages, or download anything.")
    for error in errors:
        print("ERROR: " + error)
    return not errors


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_evaluation(environment):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    experiment = f"exp_eval_aeqa_smoke_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = ROOT / "results/smoke" / experiment
    output_dir.mkdir(parents=True, exist_ok=False)
    (ROOT / ".cache/models").mkdir(parents=True, exist_ok=True)
    config_text = CONFIG.read_text()
    marker = 'exp_name: "exp_eval_aeqa_smoke"'
    if config_text.count(marker) != 1:
        raise ValueError("Smoke config must retain its original exp_name line.")
    run_config = output_dir.parent / f"{experiment}.yaml"
    run_config.write_text(config_text.replace(marker, f'exp_name: "{experiment}"'))
    command = [
        sys.executable, "run_aeqa_evaluation.py", "-cf", str(run_config.relative_to(ROOT)),
        "--start_ratio", "0", "--end_ratio", "0.006",
    ]
    print("Prashansa, --run explicitly enables downloads and Gemini requests; prompts/images leave this computer.")
    print(
        "Limits: >=30s between sends, "
        f"<={environment['GEMINI_MAX_REQUEST_ATTEMPTS']} attempts per call, "
        "<=10 sends, <=512 output tokens; stricter settings retained."
    )
    print("Configured Gemini model retained unchanged; no model or paid-tier fallback.")
    print(f"Fresh results: {output_dir}\nTimeout: 900 seconds. Ctrl+C stops the child process group.", flush=True)
    process = None
    interrupted_during_startup = False

    def defer_startup_interrupt(signum, frame):
        nonlocal interrupted_during_startup
        interrupted_during_startup = True

    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, defer_startup_interrupt)
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        if interrupted_during_startup:
            raise KeyboardInterrupt
        return_code = process.wait(timeout=900)
        if return_code:
            print("Prashansa, evaluation failed; inspect the fresh results log before retrying.")
        else:
            print("Prashansa, evaluator exited successfully; inspect results for the question outcome (not a benchmark score).")
        return return_code if return_code >= 0 else 128 - return_code
    except subprocess.TimeoutExpired:
        print("Prashansa, the 900-second limit was reached; stopping evaluation.")
        return 124
    except KeyboardInterrupt:
        print("\nPrashansa, interrupted; stopping evaluation and its child processes.")
        return 130
    finally:
        if process is not None:
            stop_group(process)


def main():
    parser = argparse.ArgumentParser(
        description="Prashansa's one-question, three-step minival smoke test. No action without an explicit flag.",
        epilog="Activate your existing 3dmem environment first. --check never downloads or calls an API; only --run evaluates.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="offline file/dependency/settings checks only; no model loading")
    modes.add_argument("--run", action="store_true", help="preflight, then explicitly allow downloads and bounded Gemini evaluation")
    arguments = parser.parse_args()
    if not (arguments.check or arguments.run):
        parser.print_help()
        return 0
    environment = child_environment()
    if not preflight(environment):
        print("Prashansa, preflight failed; no evaluation was launched. Use the documented 3dmem environment and resolve the listed items.")
        return 1
    if arguments.check:
        print("Prashansa, offline checks passed; GPU/binary compatibility and model access remain untested.")
        return 0
    return run_evaluation(environment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nPrashansa, stopped; no evaluation is continuing.")
        sys.exit(130)
    except (OSError, ValueError):
        print("Prashansa, the launcher could not complete a local operation; check file permissions and smoke configuration. No automatic retry.")
        sys.exit(1)
