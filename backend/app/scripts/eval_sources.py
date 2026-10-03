"""Before/after check for which pages a search decides to read.

    uv run python -m app.scripts.eval_sources capture
    uv run python -m app.scripts.eval_sources compare
    uv run python -m app.scripts.eval_sources run <label> [runs] [question keys...]
    uv run python -m app.scripts.eval_sources diff <label-a> <label-b>

`capture` plans and searches every control question once and stores the raw
candidates. Comparing selections on those stored candidates is the only fair
comparison: the planner writes different queries on every run and the engine
returns different pages on every call, so two live runs differ even when the
code does not.

`compare` sets the selection this branch replaced against the current one on
those stored candidates: what each would read, from which tier, and how many
of the pages are copies of a page already chosen.

`run` puts every control question through the whole pipeline and records what
reached the answer — pages read, passages kept, sources cited, names and
formulas — under a label, so a run on the old code can be set against a run on
the new. `diff` prints two such labels side by side.

Results go to backend/.eval/, which is not committed.
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from app.application.agents.constants import MAX_QUERIES, MAX_SOURCES
from app.application.agents.nodes import plan_searches_node
from app.application.services.research import ResearchService
from app.domain.search import SearchResult
from app.domain.text.similarity import same_title
from app.infrastructure.llm import create_llm_provider
from app.infrastructure.search.firecrawl import FirecrawlProvider

# Overridable so that a run on the old code, from a worktree of the base branch,
# writes next to the runs on the new.
OUT = Path(os.environ.get("EVAL_OUT") or Path(__file__).resolve().parents[2] / ".eval")

# Chosen to pull in different directions: narrow research where the scholarly
# sources are the answer, simple facts where an encyclopedia is, and practical
# engineering where the best material is documentation and blogs.
QUESTIONS: dict[str, str] = {
    "tm": "latest research on transmission matrix engineering",
    "mi": "latest research on mechanistic interpretability of transformers",
    "entropy": "What is entropy?",
    "newton": "What is Newton's second law of motion?",
    "lora": "How to fine-tune a large language model with LoRA",
    "diffusion": "how do diffusion models avoid mode collapse",
}

# Pages whose presence is the point of a question. Missing one after a change is
# a regression even if every other number improved.
LANDMARKS: dict[str, tuple[str, ...]] = {
    "tm": ("optica.org", "arxiv.org", "nature.com"),
    "mi": ("transformer-circuits.pub", "arxiv.org"),
    "entropy": ("wikipedia.org",),
    "newton": ("wikipedia.org",),
    "lora": ("huggingface.co", "arxiv.org", "sebastianraschka.com"),
    "diffusion": ("arxiv.org",),
}

CANDIDATES_PER_QUERY = 10

# The provider allows about eleven requests a minute and one question spends
# sixteen. Without a pause between questions the next one starts on an empty
# quota, and its skipped pages would be blamed on whatever code is under test.
QUOTA_RESET_SECONDS = 65

_ATTRIBUTION = re.compile(r"\b[A-Z][\w'-]+(?: et al\.?| and [A-Z][\w'-]+)? \(\d{4}\)")
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.DOTALL)
_KEPT = re.compile(r"(\d+) on topic")
_UNJUDGED = re.compile(r"(\d+) batch(?:es)? could not be judged")


def host(url: str) -> str:
    name = urlparse(url).netloc.lower()
    return name[4:] if name.startswith("www.") else name


class _FallbackSpy(logging.Handler):
    """Notices when passages were ranked by words because embedding failed —
    a run on that path is not comparable with one that embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.fired = False

    def emit(self, record: logging.LogRecord) -> None:
        if "ranking" in record.getMessage() and "instead" in record.getMessage():
            self.fired = True


def capture() -> None:
    llm = create_llm_provider()
    search = FirecrawlProvider()
    OUT.mkdir(exist_ok=True)

    for key, question in QUESTIONS.items():
        path = OUT / f"candidates-{key}.json"
        if path.exists():
            continue
        state = {"question": question, "search_query": question, "retry_count": 0}
        plan = plan_searches_node(state, llm=llm)
        per_query = []
        for query in plan["search_queries"][:MAX_QUERIES]:
            while True:
                try:
                    results = search.search(
                        query, limit=CANDIDATES_PER_QUERY, since_year=plan["since_year"]
                    )
                    break
                except Exception as exc:
                    if "rate limit" not in str(exc).lower():
                        raise
                    time.sleep(QUOTA_RESET_SECONDS)
            per_query.append([r.model_dump() for r in results])
        record = {
            "question": question,
            "queries": plan["search_queries"],
            "since_year": plan["since_year"],
            "per_query": per_query,
        }
        path.write_text(json.dumps(record, indent=2))
        print(f"{key}: {len(per_query)} queries, "
              f"{sum(len(r) for r in per_query)} candidates")


def legacy_selection(per_query: list[list[SearchResult]]) -> list[SearchResult]:
    """What search_node read before ranking: the engine's first five of each
    query, round-robin, exact URLs deduplicated, cut at MAX_SOURCES."""
    per_query = [results[:5] for results in per_query]
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for rank in range(max((len(rs) for rs in per_query), default=0)):
        for results in per_query:
            if len(merged) >= MAX_SOURCES:
                return merged
            if rank < len(results) and results[rank].url not in seen:
                seen.add(results[rank].url)
                merged.append(results[rank])
    return merged


def _copies(results: list[SearchResult]) -> int:
    """Pages whose title matches an earlier page in the same selection."""
    from app.domain.sources import title_key

    keys = [title_key(r.title) for r in results]
    return sum(1 for i, k in enumerate(keys) if any(same_title(k, j) for j in keys[:i]))


def compare() -> None:
    # Imported here, not at the top, so that `run` also works on a checkout of
    # the code from before this module existed — which is what a baseline is.
    from app.domain.sources import classify, select_sources

    for key in QUESTIONS:
        record = json.loads((OUT / f"candidates-{key}.json").read_text())
        per_query = [[SearchResult(**r) for r in rs] for rs in record["per_query"]]
        old = legacy_selection(per_query)
        new = [c.result for c in select_sources(per_query, record["question"], MAX_SOURCES).chosen]

        print(f"\n== {key}: {record['question']}")
        for label, chosen in (("old", old), ("new", new)):
            tiers = Counter(classify(r.url)[0] for r in chosen)
            print(f"  {label}: {len(chosen)} pages, {_copies(chosen)} copies, "
                  f"hosts={len({host(r.url) for r in chosen})}  "
                  + " ".join(f"{t}={tiers[t]}" for t in ("scholarly", "reference", "unknown", "low"))
                  + f"   {_landmarks(key, [r.url for r in chosen])}")
        old_urls, new_urls = {r.url for r in old}, {r.url for r in new}
        for r in new:
            mark = " " if r.url in old_urls else "+"
            print(f"    {mark} {classify(r.url)[0]:9} {host(r.url):32} {(r.title or '')[:60]}")
        for r in old:
            if r.url not in new_urls:
                print(f"    - {classify(r.url)[0]:9} {host(r.url):32} {(r.title or '')[:60]}")


def _summarise(answer) -> dict:
    steps = answer.activity
    found = [s.url for s in steps if s.kind == "source_found"]
    read = [s.url for s in steps if s.kind == "scrape_ok"]
    kept = [int(m.group(1)) for s in steps if s.kind == "grade_relevance"
            if (m := _KEPT.search(s.label))]
    cited = sorted({e.source_url for e in answer.evidence.values()})
    text = " ".join([answer.summary, answer.conclusion, *(c.text for c in answer.claims)])
    return {
        "found": found,
        "read": len(read),
        "failed": sum(1 for s in steps if s.kind == "scrape_failed"),
        "mirrors": sum(1 for s in steps if s.kind == "mirror_dropped"),
        "kept_passages": kept[-1] if kept else 0,
        "claims": len(answer.claims),
        "cited": cited,
        "cited_hosts": sorted({host(u) for u in cited}),
        "attributions": len(_ATTRIBUTION.findall(text)),
        "formulas": len(_MATH.findall(text)),
        "sufficient": answer.evidence_sufficient,
        # Kept whole so that a count that moved can be read, not just counted.
        "text": text,
        "grade_failed": sum(
            int(m.group(1)) for st in steps if st.kind == "grade_relevance"
            if (m := _UNJUDGED.search(st.detail or ""))
        ),
    }


class _TimedJudge:
    """Wraps the provider's relevance grading to record how many calls a run
    makes and how long they take — what TOP_K_CHUNKS costs."""

    def __init__(self, llm) -> None:
        self.calls: list[float] = []
        original = llm.grade_relevance

        def timed(*args, **kwargs):
            start = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                self.calls.append(time.monotonic() - start)

        llm.grade_relevance = timed


def run(label: str, runs: int, keys: list[str] | None = None) -> None:
    spy = _FallbackSpy()
    logging.getLogger("app.infrastructure.vector_store.chroma").addHandler(spy)
    llm = create_llm_provider()
    judge = _TimedJudge(llm)
    service = ResearchService(llm=llm)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"run-{label}.json"
    # Resumed rather than restarted: a full set takes most of an hour, and the
    # runs already recorded are as valid as the ones still to come.
    results: dict[str, list[dict]] = json.loads(path.read_text()) if path.exists() else {}

    for key, question in QUESTIONS.items():
        if keys and key not in keys:
            continue
        results.setdefault(key, [])
        for attempt in range(len(results[key]), runs):
            time.sleep(QUOTA_RESET_SECONDS)
            spy.fired = False
            judge.calls.clear()
            start = time.monotonic()
            try:
                summary = _summarise(service.answer(question))
            except Exception as exc:
                summary = {"error": str(exc)}
            summary["seconds"] = round(time.monotonic() - start)
            summary["grade_calls"] = len(judge.calls)
            summary["grade_max_s"] = round(max(judge.calls, default=0), 1)
            summary["grade_sum_s"] = round(sum(judge.calls), 1)
            summary["lexical_fallback"] = spy.fired
            results[key].append(summary)
            # Saved before it is printed: a run is minutes of quota, and losing
            # one to a closed stdout is not worth it.
            path.write_text(json.dumps(results, indent=2))
            print(f"{label} {key} #{attempt + 1}: "
                  + json.dumps({k: v for k, v in summary.items()
                                if k not in ("found", "cited", "text")}))


def _landmarks(key: str, urls: list[str]) -> str:
    hosts = {host(u) for u in urls}
    return " ".join(
        ("+" if any(h == m or h.endswith("." + m) for h in hosts) else "-") + m
        for m in LANDMARKS[key]
    )


def diff(a: str, b: str) -> None:
    left = json.loads((OUT / f"run-{a}.json").read_text())
    right = json.loads((OUT / f"run-{b}.json").read_text())
    fields = ("read", "failed", "mirrors", "kept_passages", "claims",
              "attributions", "formulas", "lexical_fallback",
              "grade_calls", "grade_failed", "grade_max_s", "seconds")

    for key in QUESTIONS:
        print(f"\n== {key}: {QUESTIONS[key]}")
        for label, runs in ((a, left.get(key, [])), (b, right.get(key, []))):
            for i, r in enumerate(runs):
                if "error" in r:
                    print(f"  {label}#{i + 1}  ERROR {r['error'][:80]}")
                    continue
                numbers = "  ".join(f"{f}={r[f]}" for f in fields if f in r)
                print(f"  {label}#{i + 1}  {numbers}")
                print(f"        found: {_landmarks(key, r['found'])}   "
                      f"cited: {_landmarks(key, r['cited'])}")
                print(f"        cited hosts: {', '.join(r['cited_hosts'])}")


def main() -> None:
    command, *args = sys.argv[1:] or ["help"]
    if command == "capture":
        capture()
    elif command == "compare":
        compare()
    elif command == "run" and args:
        run(args[0], int(args[1]) if len(args) > 1 else 1, args[2:] or None)
    elif command == "diff" and len(args) == 2:
        diff(*args)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
