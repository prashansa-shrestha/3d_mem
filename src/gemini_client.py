import math
import fcntl
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import openai
from openai import OpenAI

from src.const import (
    END_POINT,
    GEMINI_API_KEYS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_REQUEST_ATTEMPTS,
    GEMINI_MAX_REQUESTS_PER_RUN,
    GEMINI_MIN_REQUEST_INTERVAL,
    GEMINI_HARD_MIN_REQUEST_INTERVAL,
    GEMINI_NETWORK_TIMEOUT,
    GEMINI_ROLLING_REQUEST_LIMIT,
    GEMINI_ROLLING_WINDOW_SECONDS,
    OPENAI_KEY,
    VLM_MODEL,
)


if not OPENAI_KEY.strip():
    raise ValueError(
        "GEMINI_API_KEY_LIST and GEMINI_API_KEY are missing or blank. Set one in .env "
        "file or your environment before running evaluation."
    )

client = OpenAI(
    base_url=END_POINT, api_key=OPENAI_KEY, max_retries=0, timeout=GEMINI_NETWORK_TIMEOUT
)
_budget_path = Path(__file__).resolve().parents[1] / ".runtime" / "gemini_budget.json"
_lock = threading.Lock()
_last_start = None
_attempt_count = 0
_blocked_reason = None
_key_index = 0


def _private_file(descriptor):
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or metadata.st_nlink != 1
    ):
        raise ValueError("Unsafe budget file")


def _wall_now(ledger):
    now = time.time()
    if not math.isfinite(now) or now < ledger["updated_at"]:
        raise RuntimeError("Gemini budget clock moved backwards; no request sent.")
    ledger["updated_at"] = now
    return now


def _save_ledger(ledger):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=_budget_path.parent, prefix=".gemini-budget-", delete=False
        ) as output:
            temporary_path = output.name
            json.dump(ledger, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, _budget_path)
        temporary_path = None
        directory = os.open(_budget_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def _validate_ledger(ledger):
    # Accept the optional boot_id from the earlier implementation; it is unused.
    if not isinstance(ledger, dict) or set(ledger) - {"boot_id"} != {
        "updated_at", "attempts", "blocked_until", "last_start_monotonic", "inflight"
    }:
        raise ValueError("Invalid budget ledger")
    if type(ledger["inflight"]) is not bool:
        raise ValueError("Invalid in-flight marker")
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or len(attempts) > GEMINI_ROLLING_REQUEST_LIMIT:
        raise ValueError("Invalid budget attempts")
    for timestamp in [
        ledger["updated_at"], ledger["blocked_until"], ledger["last_start_monotonic"], *attempts
    ]:
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid budget timestamp")
    if attempts != sorted(attempts) or any(value > ledger["updated_at"] for value in attempts):
        raise ValueError("Invalid budget ordering")


@contextmanager
def _persistent_budget():
    descriptor = None
    try:
        _budget_path.parent.mkdir(mode=0o700, exist_ok=True)
        metadata = _budget_path.parent.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError("Unsafe budget directory")
        lock_path = _budget_path.with_suffix(".lock")
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            new_lock = True
        except FileExistsError:
            descriptor = os.open(lock_path, flags)
            new_lock = False
        _private_file(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            ledger_descriptor = os.open(_budget_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            if not new_lock:
                raise ValueError("Missing existing budget ledger")
            ledger = {
                "updated_at": 0, "attempts": [], "blocked_until": 0,
                "last_start_monotonic": 0, "inflight": False
            }
        else:
            with os.fdopen(ledger_descriptor) as source:
                _private_file(source.fileno())
                contents = source.read(16385)
                if len(contents) > 16384:
                    raise ValueError("Oversized budget ledger")
                ledger = json.loads(contents)
        _validate_ledger(ledger)
        if ledger["inflight"]:
            raise RuntimeError(
                "Gemini has an abandoned in-flight reservation; requests are blocked. "
                "Manual budget review is required; do not reset the ledger to bypass limits."
            )
        now = _wall_now(ledger)
        if ledger["blocked_until"] > now:
            raise RuntimeError("Gemini persistent quota cooldown is active; no request sent.")
        _save_ledger(ledger)
        yield ledger
    except (OSError, ValueError, TypeError, OverflowError):
        raise RuntimeError(
            "Gemini persistent budget state is unreadable, invalid, or unsafe; "
            "requests are blocked. Do not reset the ledger to bypass quota limits."
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _telemetry(status, wait_seconds=0.0):
    logging.info(
        "Gemini request=%d model=%s wait_seconds=%.3f status=%s",
        _attempt_count, VLM_MODEL, wait_seconds, status,
    )


def _positive_delay(value):
    try:
        delay = float(value)
    except (ValueError, TypeError, OverflowError):
        return 0.0
    return delay if math.isfinite(delay) and delay > 0 else 0.0


def _server_delay(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    if not hasattr(headers, "get"):
        headers = {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if not isinstance(retry_after, str):
        retry_after = None
    delay = _positive_delay(retry_after)
    if retry_after and not delay:
        try:
            delay = _positive_delay(
                parsedate_to_datetime(retry_after).timestamp() - time.time()
            )
        except (ValueError, TypeError, OverflowError, AttributeError, OSError):
            pass
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        payload = body.get("error", body)
        details = payload.get("details", []) if isinstance(payload, dict) else []
        if not isinstance(details, list):
            return delay
        for detail in details:
            if not isinstance(detail, dict):
                continue
            detail_type = detail.get("@type", "")
            if isinstance(detail_type, str) and detail_type.endswith("google.rpc.RetryInfo"):
                duration = detail.get("retryDelay", "")
                if isinstance(duration, str) and duration.endswith("s"):
                    delay = max(delay, _positive_delay(duration[:-1]))
    return delay


def _unavailable_quota_text(text):
    if not isinstance(text, str):
        return False
    if re.search(r"RequestsPerDay|TokensPerDay|per_day", text, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:daily|per[ -]day)\s+quota(?:\s+limit)?\s+"
        r"(?:(?:is|was|has been)\s+)?(?:exhausted|exceeded|reached|unavailable)\b"
        r"|\b(?:exhausted|exceeded|reached)\s+(?:(?:the|your)\s+)?"
        r"(?:daily|per[ -]day)\s+quota\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return bool(re.search(
        r"\blimit\s*(?::|=|\bis\b|\bof\b)\s*0(?:\.0+)?\b(?!\.\d)",
        text,
        re.IGNORECASE,
    ))


def _unavailable_quota(error):
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    payload = body.get("error", body)
    if not isinstance(payload, dict):
        return False
    message = payload.get("message", "")
    if _unavailable_quota_text(message):
        return True
    details = payload.get("details", [])
    if not isinstance(details, list):
        return False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_type = detail.get("@type", "")
        if not isinstance(detail_type, str) or not detail_type.endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations", [])
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            quota_name = " ".join(
                str(violation.get(field, ""))
                for field in ("quotaMetric", "quotaId", "description")
            )
            if _unavailable_quota_text(quota_name):
                return True
            if any(
                str(violation.get(field, "")).strip() in ("0", "0.0")
                for field in ("quotaValue", "quotaLimit", "limit")
            ):
                return True
    return False


def request_completion(messages) -> Optional[str]:
    global _last_start, _attempt_count, _blocked_reason

    with _lock:
        if _blocked_reason:
            raise RuntimeError(_blocked_reason)
        with _persistent_budget() as ledger:
            return _request_with_budget(messages, ledger)


def _finish_attempt(ledger):
    now = _wall_now(ledger)
    ledger["attempts"][-1] = now
    ledger["last_start_monotonic"] = time.monotonic()
    ledger["inflight"] = False
    return now


def _request_with_budget(messages, ledger):
    global _last_start, _attempt_count, _blocked_reason, _key_index

    try:
        retry_not_before = 0.0
        attempt_index = 0
        while attempt_index < GEMINI_MAX_REQUEST_ATTEMPTS:
            if _attempt_count >= GEMINI_MAX_REQUESTS_PER_RUN:
                _blocked_reason = (
                    "Gemini per-process request safety cap reached "
                    "(GEMINI_MAX_REQUESTS_PER_RUN). Check your project quota before restarting."
                )
                raise RuntimeError(_blocked_reason)
            now = _wall_now(ledger)
            ledger["attempts"] = [
                timestamp for timestamp in ledger["attempts"]
                if timestamp > now - GEMINI_ROLLING_WINDOW_SECONDS
            ]
            if len(ledger["attempts"]) >= GEMINI_ROLLING_REQUEST_LIMIT:
                _blocked_reason = "Gemini hard rolling-24-hour cap of 20 attempts reached; no request sent."
                raise RuntimeError(_blocked_reason)
            interval = max(GEMINI_HARD_MIN_REQUEST_INTERVAL, GEMINI_MIN_REQUEST_INTERVAL)
            next_start = retry_not_before
            if _last_start is not None:
                next_start = max(next_start, _last_start + interval)
            if ledger["attempts"]:
                # Persisted wall time works across reboots; monotonic time does not.
                next_start = max(next_start, time.monotonic() + max(
                    0, ledger["attempts"][-1] + interval - now
                ))
            while True:
                _wall_now(ledger)
                remaining = next_start - time.monotonic()
                if remaining <= 0:
                    break
                _telemetry("waiting", remaining)
                time.sleep(remaining)
            now = _wall_now(ledger)
            ledger["attempts"].append(now)
            ledger["last_start_monotonic"] = time.monotonic()
            ledger["inflight"] = True
            _save_ledger(ledger)
            _last_start = time.monotonic()
            _attempt_count += 1
            _telemetry("sending")
            try:
                completion = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    top_p=0.95,
                )
            except openai.APIError as error:
                error_message = str(error)
                for key in GEMINI_API_KEYS:
                    error_message = error_message.replace(key, "[redacted]")
                logging.error("Gemini request failed: %s", error_message)
                now = _finish_attempt(ledger)
                status = getattr(error, "status_code", None)
                _telemetry(status if isinstance(status, int) else "connection_or_api_error")
                if status == 429 and _unavailable_quota(error):
                    if _key_index + 1 < len(GEMINI_API_KEYS):
                        _key_index += 1
                        client.api_key = GEMINI_API_KEYS[_key_index]
                        logging.warning(
                            "Gemini quota exhausted; switching to key %d of %d",
                            _key_index + 1, len(GEMINI_API_KEYS),
                        )
                        _save_ledger(ledger)
                        retry_not_before = time.monotonic() + _server_delay(error)
                        continue
                    ledger["blocked_until"] = now + max(
                        GEMINI_ROLLING_WINDOW_SECONDS, _server_delay(error)
                    )
                    _save_ledger(ledger)
                    _blocked_reason = (
                        "Gemini reports quota exhaustion for all configured keys. "
                        "Further requests are blocked in this process; check project quota "
                        "and model availability before restarting."
                    )
                    raise RuntimeError(_blocked_reason) from None
                temporary = (
                    isinstance(error, openai.APIConnectionError)
                    or status == 429
                    or (isinstance(status, int) and 500 <= status < 600)
                )
                if not temporary:
                    _save_ledger(ledger)
                    _blocked_reason = (
                        "Gemini request failed with a non-retryable API error. "
                        "Check credentials, model access, and request configuration."
                    )
                    raise RuntimeError(_blocked_reason) from None
                final_attempt = (
                    attempt_index + 1 >= GEMINI_MAX_REQUEST_ATTEMPTS
                    or _attempt_count >= GEMINI_MAX_REQUESTS_PER_RUN
                    or len(ledger["attempts"]) >= GEMINI_ROLLING_REQUEST_LIMIT
                )
                delay = max(60 if final_attempt else 30 * 2 ** attempt_index, _server_delay(error))
                ledger["blocked_until"] = max(ledger["blocked_until"], now + delay)
                _save_ledger(ledger)
                if final_attempt:
                    _blocked_reason = (
                        "Gemini rate-limit or temporary failure retries exhausted. "
                        "Further requests are blocked in this process; check project quota "
                        "and service availability before restarting."
                    )
                    raise RuntimeError(_blocked_reason) from None
                if _attempt_count < GEMINI_MAX_REQUESTS_PER_RUN:
                    retry_not_before = time.monotonic() + delay
                attempt_index += 1
                continue
            _finish_attempt(ledger)
            _save_ledger(ledger)
            _telemetry("success")
            if not completion.choices:
                return None
            return completion.choices[0].message.content or None
    except RuntimeError:
        if not _blocked_reason:
            _blocked_reason = "Gemini request guard stopped; further requests are blocked in this process."
        raise
