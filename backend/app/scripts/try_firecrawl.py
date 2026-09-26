from app.infrastructure.search.firecrawl import FirecrawlProvider
from app.core.config import get_settings
from firecrawl import FirecrawlApp

settings = get_settings()
client = FirecrawlApp(api_key=settings.firecrawl_api_key)

response = client.scrape("https://en.wikipedia.org/wiki/Gradient_descent", formats=["markdown"])
print(type(response))
print(response.markdown[:300])