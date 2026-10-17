import math
import re
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Optional

import openai
from openai import OpenAI

from src.const import (
    END_POINT,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_REQUEST_ATTEMPTS,
    GEMINI_MAX_REQUESTS_PER_RUN,
    GEMINI_MIN_REQUEST_INTERVAL,
    OPENAI_KEY,
    VLM_MODEL,
)


if not OPENAI_KEY.strip():
    raise ValueError(
        "GEMINI_API_KEY is missing or blank. Set it in the repository-root .env "
        "file or your environment before running evaluation."
    )

client = OpenAI(base_url=END_POINT, api_key=OPENAI_KEY, max_retries=0)
_lock = threading.Lock()
_last_start = None
_attempt_count = 0
_blocked_reason = None


def _positive_delay(value):
    try:
        delay = float(value)
    except (ValueError, TypeError, OverflowError):
        return 0.0
    return delay if math.isfinite(delay) and delay > 0 else 0.0


def _server_delay(error):
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    delay = _positive_delay(retry_after)
    if retry_after and not delay:
        try:
            delay = _positive_delay(
                parsedate_to_datetime(retry_after).timestamp() - time.time()
            )
        except (ValueError, TypeError, OverflowError):
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
        retry_not_before = 0.0
        for attempt_index in range(GEMINI_MAX_REQUEST_ATTEMPTS):
            if _attempt_count >= GEMINI_MAX_REQUESTS_PER_RUN:
                _blocked_reason = (
                    "Gemini per-process request safety cap reached "
                    "(GEMINI_MAX_REQUESTS_PER_RUN). Check your project quota before restarting."
                )
                raise RuntimeError(_blocked_reason)
            next_start = retry_not_before
            if _last_start is not None:
                next_start = max(next_start, _last_start + GEMINI_MIN_REQUEST_INTERVAL)
            while True:
                remaining = next_start - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(remaining)
            _last_start = time.monotonic()
            _attempt_count += 1
            try:
                completion = client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    top_p=0.95,
                )
            except openai.APIError as error:
                status = getattr(error, "status_code", None)
                if status == 429 and _unavailable_quota(error):
                    _blocked_reason = (
                        "Gemini reports a daily quota exhaustion or zero quota limit. "
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
                    _blocked_reason = (
                        "Gemini request failed with a non-retryable API error. "
                        "Check credentials, model access, and request configuration."
                    )
                    raise RuntimeError(_blocked_reason) from None
                if attempt_index + 1 >= GEMINI_MAX_REQUEST_ATTEMPTS:
                    _blocked_reason = (
                        "Gemini rate-limit or temporary failure retries exhausted. "
                        "Further requests are blocked in this process; check project quota "
                        "and service availability before restarting."
                    )
                    raise RuntimeError(_blocked_reason) from None
                if _attempt_count < GEMINI_MAX_REQUESTS_PER_RUN:
                    delay = max(30 * 2 ** attempt_index, _server_delay(error))
                    retry_not_before = time.monotonic() + delay
                continue
            if not completion.choices:
                return None
            return completion.choices[0].message.content or None
