class ResearchServiceError(Exception):
    """Base error for the research pipeline."""


class UpstreamServiceError(ResearchServiceError):
    """Raised when an external provider (Firecrawl, Gemini) fails
    in a way the pipeline cannot recover from on its own."""