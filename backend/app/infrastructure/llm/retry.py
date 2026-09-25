"""One retry policy for every language model provider.

The two providers used to carry a marker list each, and the lists disagreed:
Gemini's knew RESOURCE_EXHAUSTED but not "overloaded", OpenRouter's the reverse,
so the same outage was retried on one and surfaced on the other. Vendors differ
in wording, not in which failures are worth another attempt.
"""

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

MAX_ATTEMPTS = 3

_TRANSIENT_MARKERS = (
    "429",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "unavailable",
    "overloaded",
    "rate limit",
    "timeout",
    "timed out",
)


def is_transient_error(exc: BaseException) -> bool:
    """True when the failure is worth retrying rather than reporting."""
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


llm_retry = retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
