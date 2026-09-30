class ResearchServiceError(Exception):
    """Base error for the research pipeline."""


class UpstreamServiceError(ResearchServiceError):
    """Raised when an external provider (Firecrawl, Gemini) fails
    in a way the pipeline cannot recover from on its own."""


class ProviderCreditsExhausted(UpstreamServiceError):
    """The language model provider refused a request for lack of credit.

    Retrying cannot help and neither can searching again, and swallowing it is
    worse than either: every call of the run fails the same way, and the reader
    was told that no reliable sources exist when the sources were fine.
    """
