import os
import math
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# about habitat scene
INVALID_SCENE_ID = []

END_POINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_KEY = GEMINI_API_KEY
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


GEMINI_MIN_REQUEST_INTERVAL = _positive_setting("GEMINI_MIN_REQUEST_INTERVAL", 15, float)
GEMINI_MAX_REQUEST_ATTEMPTS = _positive_setting("GEMINI_MAX_REQUEST_ATTEMPTS", 3, int)
GEMINI_MAX_REQUESTS_PER_RUN = _positive_setting("GEMINI_MAX_REQUESTS_PER_RUN", 20, int)
GEMINI_MAX_OUTPUT_TOKENS = _positive_setting("GEMINI_MAX_OUTPUT_TOKENS", 512, int)
