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
    # OpenRouter reserves credit for every request in flight; a refusal on that
    # ground clears as soon as the others finish.
    "in-flight requests",
    # A reply that never called the output tool; the next attempt usually does.
    "no structured output",
)

# A refusal to spend: the balance cannot cover the request. Checked before the
# transient markers, because no amount of waiting fixes it.
_BILLING_MARKERS = (
    "error code: 402",
    "payment required",
    "requires more credits",
    "insufficient credits",
)


def is_billing_error(exc: BaseException) -> bool:
    """True when the provider refused the request because it cannot be paid for.

    A refusal over requests already in flight is not one: it clears on its own.
    """
    message = str(exc).lower()
    if "in-flight requests" in message:
        return False
    return any(marker in message for marker in _BILLING_MARKERS)


def is_transient_error(exc: BaseException) -> bool:
    """True when the failure is worth retrying rather than reporting."""
    if is_billing_error(exc):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


llm_retry = retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
