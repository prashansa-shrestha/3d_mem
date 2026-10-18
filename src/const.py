import os
import math
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# about habitat scene
INVALID_SCENE_ID = []

END_POINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Accept [key1, key2] and ["key1", "key2"] without evaluating .env contents.
GEMINI_API_KEYS = list(dict.fromkeys(
    key.strip().strip("\"'").strip()
    for key in os.getenv("GEMINI_API_KEY_LIST", "").strip().strip("[]").split(",")
    if key.strip().strip("\"'").strip()
))
if not GEMINI_API_KEYS and GEMINI_API_KEY.strip():
    GEMINI_API_KEYS = [GEMINI_API_KEY.strip()]
OPENAI_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""
VLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def _positive_setting(name, default, converter):
    try:
        value = converter(os.getenv(name, str(default)))
        if value <= 0 or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError
    except (ValueError, OverflowError):
        kind = "finite positive number" if converter is float else "positive integer"
        raise ValueError(f"{name} must be a {kind}.") from None
    return value


GEMINI_REQUESTS_PER_MINUTE = _positive_setting("GEMINI_REQUESTS_PER_MINUTE", 7, int)
GEMINI_REQUESTS_PER_DAY = _positive_setting("GEMINI_REQUESTS_PER_DAY", 1200, int)
GEMINI_HARD_MIN_REQUEST_INTERVAL = 60 / GEMINI_REQUESTS_PER_MINUTE
GEMINI_MIN_REQUEST_INTERVAL = _positive_setting(
    "GEMINI_MIN_REQUEST_INTERVAL", GEMINI_HARD_MIN_REQUEST_INTERVAL, float
)
GEMINI_MAX_REQUEST_ATTEMPTS = _positive_setting("GEMINI_MAX_REQUEST_ATTEMPTS", 3, int)
GEMINI_MAX_REQUESTS_PER_RUN = _positive_setting(
    "GEMINI_MAX_REQUESTS_PER_RUN", GEMINI_REQUESTS_PER_DAY, int
)
GEMINI_MAX_OUTPUT_TOKENS = _positive_setting("GEMINI_MAX_OUTPUT_TOKENS", 512, int)
GEMINI_ROLLING_REQUEST_LIMIT = GEMINI_REQUESTS_PER_DAY
GEMINI_ROLLING_WINDOW_SECONDS = 24 * 60 * 60
GEMINI_NETWORK_TIMEOUT = 60
