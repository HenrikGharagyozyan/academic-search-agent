import type { AnswerEvidence } from "../types/answer";

/** The host without "www.", or the raw string if it is not a URL. */
export function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

/**
 * How a reader knows these sites. A host is a poor name — "opg.optica.org",
 * "pmc.ncbi.nlm.nih.gov" — and the page title is the article, not the site,
 * so the common ones are named here and the rest fall back to their domain.
 */
const SITE_NAMES: Record<string, string> = {
  "arxiv.org": "arXiv",
  "biorxiv.org": "bioRxiv",
  "medrxiv.org": "medRxiv",
  "openreview.net": "OpenReview",
  "aclanthology.org": "ACL Anthology",
  "proceedings.mlr.press": "PMLR",
  "neurips.cc": "NeurIPS",
  "nature.com": "Nature",
  "science.org": "Science",
  "sciencedirect.com": "ScienceDirect",
  "springer.com": "Springer",
  "wiley.com": "Wiley",
  "tandfonline.com": "Taylor & Francis",
  "ieee.org": "IEEE",
  "acm.org": "ACM",
  "aps.org": "APS",
  "optica.org": "Optica",
  "aip.org": "AIP",
  "iop.org": "IOP",
  "plos.org": "PLOS",
  "frontiersin.org": "Frontiers",
  "mdpi.com": "MDPI",
  "pnas.org": "PNAS",
  "cell.com": "Cell",
  "cambridge.org": "Cambridge",
  "oup.com": "Oxford Academic",
  "ncbi.nlm.nih.gov": "PubMed",
  "semanticscholar.org": "Semantic Scholar",
  "researchgate.net": "ResearchGate",
  "wikipedia.org": "Wikipedia",
  "britannica.com": "Britannica",
  "libretexts.org": "LibreTexts",
  "openstax.org": "OpenStax",
  "stackexchange.com": "Stack Exchange",
  "github.com": "GitHub",
  "huggingface.co": "Hugging Face",
  "transformer-circuits.pub": "Transformer Circuits",
  "distill.pub": "Distill",
  "lesswrong.com": "LessWrong",
  "medium.com": "Medium",
  "substack.com": "Substack",
  "reddit.com": "Reddit",
  "youtube.com": "YouTube",
};

/** "pmc.ncbi.nlm.nih.gov" → "PubMed"; an unlisted host keeps its domain. */
export function siteName(url: string): string {
  const host = hostOf(url);
  const labels = host.split(".");
  for (let i = 0; i < labels.length - 1; i++) {
    const name = SITE_NAMES[labels.slice(i).join(".")];
    if (name) return name;
  }
  return host;
}

/**
 * The site's icon. Fetched from DuckDuckGo's icon service by host, so the
 * request carries the source's domain but nothing about the question.
 */
export function faviconUrl(url: string): string {
  return `https://icons.duckduckgo.com/ip3/${hostOf(url)}.ico`;
}

/** The passages a claim cites, one group per page, in the order first cited. */
export function groupBySource(passages: AnswerEvidence[]): AnswerEvidence[][] {
  const groups = new Map<string, AnswerEvidence[]>();
  for (const passage of passages) {
    const group = groups.get(passage.source_url);
    if (group) group.push(passage);
    else groups.set(passage.source_url, [passage]);
  }
  return [...groups.values()];
}
