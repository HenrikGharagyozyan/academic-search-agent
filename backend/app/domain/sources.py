"""Choosing which search results to read, before any of them is read.

Taking the engine's results in order read whatever happened to rank first —
Reddit for "what is entropy", Medium for LoRA — while the journal article sat at
position seven of another query. Only twelve pages are read, so which twelve
decides what the answer can say.

Three things are known about a result before it is scraped: where it lives,
what its title and snippet say, and where it ranked. Each says something
different. The host says whether the page is scholarly; the words say whether it
is about the question; the rank says what the engine thought of it. None is
enough alone, and scholarliness is deliberately not allowed to outweigh the
other two: an arXiv paper that only grazes the subject is worth less than a
documentation page or a well-known blog that answers it, and for a practical
question that is most of the good material.

Low-priority hosts are demoted, never excluded. A forum thread is sometimes the
only place a thing is written down, and when nothing better turned up it should
still be read.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlparse

from app.domain.query import anchor_sequence
from app.domain.search import SearchResult
from app.domain.text.plain import to_label
from app.domain.text.similarity import same_title, title_words

Tier = Literal["scholarly", "reference", "unknown", "low"]

TIER_WEIGHT: dict[Tier, float] = {
    "scholarly": 1.0,
    "reference": 0.65,
    # An unrecognised host is neutral, not suspect. Most of the web is
    # unrecognised, and the good engineering write-ups are disproportionately
    # on personal domains no list will ever name.
    "unknown": 0.5,
    "low": 0.2,
}

# Publishers, preprint servers, proceedings, indexes and research labs.
SCHOLARLY_HOSTS = frozenset(
    """
    arxiv.org biorxiv.org medrxiv.org chemrxiv.org engrxiv.org psyarxiv.com
    osf.io ssrn.com preprints.org hal.science openreview.net aclanthology.org
    proceedings.mlr.press jmlr.org neurips.cc papers.nips.cc proceedings.neurips.cc
    icml.cc iclr.cc aaai.org ijcai.org cvf.com thecvf.com usenix.org
    nature.com science.org cell.com pnas.org sciencedirect.com springer.com
    link.springer.com wiley.com onlinelibrary.wiley.com tandfonline.com
    sagepub.com ieeexplore.ieee.org ieee.org acm.org dl.acm.org aps.org
    journals.aps.org link.aps.org iop.org iopscience.iop.org optica.org
    opg.optica.org spiedigitallibrary.org aip.org pubs.aip.org acs.org
    pubs.acs.org rsc.org pubs.rsc.org academic.oup.com cambridge.org
    annualreviews.org royalsocietypublishing.org plos.org journals.plos.org
    frontiersin.org mdpi.com hindawi.com elifesciences.org biomedcentral.com
    jstor.org scielo.br scielo.org informs.org pubsonline.informs.org
    siam.org ams.org projecteuclid.org worldscientific.com degruyter.com
    ncbi.nlm.nih.gov pubmed.ncbi.nlm.nih.gov pmc.ncbi.nlm.nih.gov europepmc.org
    doi.org zenodo.org
    transformer-circuits.pub distill.pub deepmind.google deepmind.com
    research.google ai.googleblog.com research.ibm.com machinelearning.apple.com
    disneyresearch.com mpg.de cnrs.fr inria.fr cern.ch
    """.split()
)

# Hosts where research and marketing share a domain: only the research path counts.
SCHOLARLY_PATHS: tuple[tuple[str, str], ...] = (
    ("microsoft.com", "/en-us/research"),
    ("ai.meta.com", "/research"),
    ("openai.com", "/research"),
    ("anthropic.com", "/research"),
    ("nvidia.com", "/research"),
)

# Encyclopedias, textbooks, documentation, paper aggregators, and blogs that
# practitioners read as primary sources.
REFERENCE_HOSTS = frozenset(
    """
    wikipedia.org britannica.com plato.stanford.edu mathworld.wolfram.com
    libretexts.org openstax.org scholarpedia.org nlab-pages.org
    semanticscholar.org scholar.google.com researchgate.net mendeley.com
    inspirehep.net paperswithcode.com alphaxiv.org emergentmind.com
    huggingface.co pytorch.org tensorflow.org jax.readthedocs.io keras.io
    scikit-learn.org readthedocs.io docs.python.org developer.nvidia.com
    learn.microsoft.com cloud.google.com docs.aws.amazon.com github.com
    lilianweng.github.io sebastianraschka.com colah.github.io jalammar.github.io
    karpathy.github.io thegradient.pub gradientscience.org lightning.ai
    neuronpedia.org lesswrong.com alignmentforum.org
    physics.stackexchange.com math.stackexchange.com stats.stackexchange.com
    mathoverflow.net
    """.split()
)

# Open publishing platforms, video, social networks, homework and content farms.
LOW_HOSTS = frozenset(
    """
    reddit.com youtube.com youtu.be medium.com towardsdatascience.com
    towardsai.net substack.com quora.com linkedin.com facebook.com x.com
    twitter.com instagram.com tiktok.com pinterest.com dev.to hashnode.dev
    geeksforgeeks.org analyticsvidhya.com scribd.com studocu.com coursehero.com
    slideshare.net chegg.com byjus.com study.com cliffsnotes.com brainly.com
    """.split()
)

_ACADEMIC_HOST = re.compile(
    r"(?:^|\.)(?:edu|ac\.[a-z]{2}|edu\.[a-z]{2}|gov|gov\.[a-z]{2})$"
    r"|(?:^|\.)(?:uni|univ)-[a-z-]+\.|universit|(?:^|\.)research\."
)

# A page that looks like a paper wherever it is hosted.
_PAPER_PATH = re.compile(
    r"/(?:doi|abs|pdf|article|articles|paper|papers|proceedings|journal|journals|"
    r"volume|issue|publication|publications)(?:/|$)|\.pdf$",
    re.IGNORECASE,
)
_PAPER_TEXT = re.compile(
    r"\b10\.\d{4,9}/\S+|\barxiv:\s?\d{4}\.\d{4,5}|\bjournal of\b|\bproceedings of\b"
    r"|\bvol\.\s?\d|\bet al\b|\bpeer[- ]reviewed\b|\bissn\b|^\s*\[pdf\]",
    re.IGNORECASE,
)
PAPER_SIGNAL_BONUS = 0.1
# An unknown host never reaches the scholarly weight on signals alone: a blog
# that cites "et al." and links a PDF is still a blog.
PAPER_SIGNAL_CAP = 0.3

# score = scholarly·S + relevance·R + rank·P. Scholarliness leads, but the gap
# between the top and bottom tiers (0.32) is less than what relevance alone can
# move (0.35), so an on-topic page from anywhere beats an off-topic paper.
WEIGHT_SCHOLARLY = 0.40
WEIGHT_RELEVANCE = 0.35
WEIGHT_RANK = 0.25

# Reciprocal-rank fusion constant. Results found by several queries add up, so
# a page every angle turned up outranks one that a single query ranked first.
RRF_K = 10

# Applied per page already taken from the same host, so a fifth arXiv paper has
# to be clearly better than the best page from somewhere else. Mild on purpose:
# four arXiv papers are four papers.
HOST_PENALTY = 0.15
# A page whose title matches one already taken is almost certainly the same
# paper on another site. Large enough to put it behind every distinct page, not
# so large that it is never read when nothing else is left.
MIRROR_PENALTY = 0.6

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_DOI = re.compile(r"(10\.\d{4,9}/[^?#\s]+?)(?:\.pdf|/full|/abstract|/pdf)?/?$", re.IGNORECASE)
_TRACKING = re.compile(r"^(?:utm_|fbclid$|gclid$|ref$|ref_src$)")
# "[PDF] Title", "(PDF) Title", "[2410.08315] Title" — prefixes that mirrors add
# and the original does not.
_TITLE_PREFIX = re.compile(r"^\s*(?:\[[^\]]{1,20}\]|\([^)]{1,10}\))\s*")
# "Title - Wikipedia", "Title — Site Name": a trailing site name is short.
_TITLE_SUFFIX = re.compile(r"\s+[-–—]\s+(?:\S+\s?){1,4}$")


def host_of(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _listed(host: str, hosts: frozenset[str]) -> str | None:
    """The entry of ``hosts`` the host falls under, most specific first."""
    labels = host.split(".")
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        if candidate in hosts:
            return candidate
    return None


def _scholarly_path(host: str, path: str) -> bool:
    return any(
        (host == domain or host.endswith("." + domain)) and path.startswith(prefix)
        for domain, prefix in SCHOLARLY_PATHS
    )


def classify(url: str) -> tuple[Tier, str]:
    """The host's tier, and the name it is grouped under for spreading.

    The most specific listing wins, so "magazine.sebastianraschka.com" is the
    blog and "lush93md.medium.com" is Medium.
    """
    host = host_of(url)
    path = urlparse(url).path.lower()

    if _scholarly_path(host, path):
        return "scholarly", host

    matches = [
        (entry, tier)
        for tier, hosts in (
            ("scholarly", SCHOLARLY_HOSTS),
            ("reference", REFERENCE_HOSTS),
            ("low", LOW_HOSTS),
        )
        if (entry := _listed(host, hosts)) is not None
    ]
    if matches:
        entry, tier = max(matches, key=lambda m: m[0].count("."))
        return tier, entry

    if _ACADEMIC_HOST.search(host):
        return "scholarly", host
    return "unknown", host


def scholarly_weight(result: SearchResult) -> tuple[Tier, float]:
    tier, _ = classify(result.url)
    weight = TIER_WEIGHT[tier]
    if tier != "unknown":
        return tier, weight

    signals = bool(_PAPER_PATH.search(urlparse(result.url).path))
    signals += len(_PAPER_TEXT.findall(f"{result.title}\n{result.snippet}"))
    return tier, weight + min(PAPER_SIGNAL_CAP, PAPER_SIGNAL_BONUS * signals)


_SUFFIXES = ("ing", "ed", "es", "e", "s")


def _stem(word: str) -> str:
    """Enough to make "tune", "tuned" and "tuning" one word, and no more."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _initials(words: list[str]) -> dict[str, range]:
    """Every run of two to four consecutive words, keyed by its initials, when
    those are long enough not to collide with an ordinary word by accident."""
    runs: dict[str, range] = {}
    for size in (2, 3, 4):
        for start in range(len(words) - size + 1):
            letters = "".join(w[0] for w in words[start : start + size])
            if len(letters) >= 3:
                runs.setdefault(letters, range(start, start + size))
    return runs


def _covered(question: list[str], text: str) -> set[int]:
    """Which of the question's words the text names, allowing for the ways a
    title says the same thing differently: "finetuning" for "fine-tune", "LLMs"
    for "large language model", and the reverse.

    Measured on a captured search: matching words exactly scored "Practical Tips
    for Finetuning LLMs Using LoRA" as barely about "how to fine-tune a large
    language model with LoRA", and it was not read.
    """
    words = anchor_sequence(text)
    stems = {_stem(w) for w in words}
    covered = {i for i, w in enumerate(question) if _stem(w) in stems}

    for i in range(len(question) - 1):
        joined = question[i] + _stem(question[i + 1])
        if any(_stem(w).startswith(joined) for w in words):
            covered |= {i, i + 1}

    present = set(words)
    for letters, run in _initials(question).items():
        if letters in present:
            covered |= set(run)

    spelled_out = _initials(words)
    covered |= {i for i, w in enumerate(question) if w in spelled_out}
    return covered


def relevance(result: SearchResult, question: list[str]) -> float:
    """How much of the question's subject the title and snippet name.

    The title counts for more: it is what the page is about, while a snippet is
    often a passage that merely mentions the term.
    """
    if not question:
        return 1.0
    title = to_label(result.title) or result.title
    in_title = _covered(question, title)
    in_either = in_title | _covered(question, result.snippet)
    return 0.6 * len(in_title) / len(question) + 0.4 * len(in_either) / len(question)


def canonical_url(url: str) -> str:
    """One key per document: arXiv's abs, pdf and versions are one paper, as are
    a DOI's landing and PDF pages, and tracking parameters change nothing."""
    parsed = urlparse(url)
    host = host_of(url)
    path = parsed.path

    if host.endswith("arxiv.org") and (match := _ARXIV_ID.search(path)):
        return f"arxiv:{match.group(1)}"
    if match := _DOI.search(path):
        return f"doi:{match.group(1).lower()}"

    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parsed.query) if not _TRACKING.match(k)
    ))
    return f"{host}{path.rstrip('/')}" + (f"?{query}" if query else "")


def title_key(title: str) -> frozenset[str]:
    """The words identifying a document, with the decoration mirrors add removed."""
    label = to_label(title) or title
    label = _TITLE_PREFIX.sub("", label)
    label = _TITLE_SUFFIX.sub("", label).rstrip(" .…")
    return title_words(label)


@dataclass
class Candidate:
    result: SearchResult
    key: str
    queries: set[int] = field(default_factory=set)
    fusion: float = 0.0
    tier: Tier = "unknown"
    score: float = 0.0
    group: str = ""
    title: frozenset[str] = frozenset()


@dataclass
class SourceSelection:
    chosen: list[Candidate]
    considered: int
    # Candidates left out because a page with the same title was chosen.
    mirrors_set_aside: int


def _merge(per_query: list[list[SearchResult]]) -> list[Candidate]:
    by_key: dict[str, Candidate] = {}
    for q, results in enumerate(per_query):
        for rank, result in enumerate(results):
            key = canonical_url(result.url)
            candidate = by_key.setdefault(key, Candidate(result=result, key=key))
            if q not in candidate.queries:
                candidate.queries.add(q)
                candidate.fusion += 1 / (RRF_K + rank)
    return list(by_key.values())


def agreement(candidate: Candidate, queries: int) -> float:
    """How many of the other queries found this page too, from 0 to 1.

    The second measure of relevance, for the page whose title is too terse to
    match: every planned query has passed the check that it stays on the
    subject, so a page most of them returned is about the subject whatever its
    title says. Hugging Face's "LoRA (Low-Rank Adaptation)" names one word of
    "how to fine-tune a large language model with LoRA", and three of four
    queries found it. With a single query there is nothing to agree with.
    """
    if queries < 2:
        return 0.0
    return (len(candidate.queries) - 1) / (queries - 1)


def _score(candidates: list[Candidate], question: str, queries: int) -> None:
    # Each subject word once, in the question's order.
    terms = list(dict.fromkeys(anchor_sequence(question)))
    best_fusion = max(c.fusion for c in candidates)
    for c in candidates:
        c.tier, weight = scholarly_weight(c.result)
        c.group = classify(c.result.url)[1]
        c.title = title_key(c.result.title)
        c.score = (
            WEIGHT_SCHOLARLY * weight
            + WEIGHT_RELEVANCE * max(relevance(c.result, terms), agreement(c, queries))
            + WEIGHT_RANK * c.fusion / best_fusion
        )


def _is_mirror(candidate: Candidate, taken: list[Candidate]) -> bool:
    return any(same_title(candidate.title, t.title) for t in taken)


def select_sources(
    per_query: list[list[SearchResult]], question: str, limit: int
) -> SourceSelection:
    """The ``limit`` results most worth reading, best first.

    Each query's best result is taken first, as the round-robin that this
    replaced did: the queries were planned to pull in different directions, and
    a ranking alone would let the one closest to the question take every slot.
    The rest are taken greedily, each pick paying for every page already taken
    from its host and for matching the title of one already taken.
    """
    candidates = _merge(per_query)
    if not candidates or limit <= 0:
        return SourceSelection(chosen=[], considered=len(candidates), mirrors_set_aside=0)
    _score(candidates, question, len(per_query))

    taken: list[Candidate] = []
    per_group: Counter[str] = Counter()

    def adjusted(c: Candidate) -> float:
        return (
            c.score
            - HOST_PENALTY * per_group[c.group]
            - (MIRROR_PENALTY if _is_mirror(c, taken) else 0.0)
        )

    def take(c: Candidate) -> None:
        taken.append(c)
        per_group[c.group] += 1

    for q in range(len(per_query)):
        if len(taken) >= limit:
            break
        own = [c for c in candidates if q in c.queries and c not in taken]
        if own:
            take(max(own, key=adjusted))

    while len(taken) < limit:
        remaining = [c for c in candidates if c not in taken]
        if not remaining:
            break
        take(max(remaining, key=adjusted))

    left_out = [c for c in candidates if c not in taken]
    return SourceSelection(
        chosen=sorted(taken, key=lambda c: c.score, reverse=True),
        considered=len(candidates),
        mirrors_set_aside=sum(1 for c in left_out if _is_mirror(c, taken)),
    )
