"""Which search results are read: scholarly first, but never at any price.

The cases come from captured searches. "What is entropy?" put Reddit first and
Wikipedia second; LoRA put Medium first and Hugging Face sixth; one transmission
matrix paper came back from eight hosts across four queries.
"""

import pytest

from app.domain.search import SearchResult
from app.domain.sources import (
    canonical_url,
    classify,
    relevance,
    scholarly_weight,
    select_sources,
    title_key,
)
from app.domain.query import anchor_sequence


def r(url: str, title: str = "", snippet: str = "") -> SearchResult:
    return SearchResult(title=title, url=url, snippet=snippet)


# --- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "url, tier",
    [
        ("https://arxiv.org/abs/2410.08315", "scholarly"),
        ("https://opg.optica.org/oe/fulltext.cfm?uri=oe-1", "scholarly"),
        ("https://transformer-circuits.pub/2025/attribution-graphs/", "scholarly"),
        ("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/", "scholarly"),
        ("https://diffusion.csail.mit.edu/notes.pdf", "scholarly"),
        ("https://yaronbromberg.huji.ac.il/paper", "scholarly"),
        ("https://www.microsoft.com/en-us/research/blog/x", "scholarly"),
        ("https://en.wikipedia.org/wiki/Entropy", "reference"),
        ("https://magazine.sebastianraschka.com/p/lora", "reference"),
        ("https://huggingface.co/docs/peft/lora", "reference"),
        ("https://www.lesswrong.com/posts/x", "reference"),
        ("https://www.neuronpedia.org/circuits", "reference"),
        ("https://www.reddit.com/r/Physics/x", "low"),
        ("https://lush93md.medium.com/lora", "low"),
        ("https://pub.towardsai.net/x", "low"),
        ("https://www.youtube.com/watch?v=1", "low"),
        ("https://zohaib.me/lora-guide", "unknown"),
        ("https://www.microsoft.com/en-us/windows", "unknown"),
    ],
)
def test_hosts_fall_into_their_tier(url, tier):
    assert classify(url)[0] == tier


def test_a_subdomain_groups_under_its_listed_host():
    assert classify("https://lush93md.medium.com/a")[1] == "medium.com"
    assert classify("https://medium.com/b")[1] == "medium.com"


def test_paper_signals_lift_an_unknown_host_but_not_to_scholarly():
    plain = scholarly_weight(r("https://lab.example.org/notes"))[1]
    paper = scholarly_weight(r(
        "https://www.institut-langevin.espci.fr/biblio/paper.pdf",
        title="[PDF] Wavefront shaping in multimode fibers",
        snippet="Journal of Optics, Vol. 12 — doi 10.1364/OE.1 et al.",
    ))[1]
    assert plain == 0.5
    assert 0.5 < paper < 1.0


def test_paper_signals_do_not_rescue_a_low_host():
    assert scholarly_weight(r("https://medium.com/x.pdf", snippet="et al. doi 10.1/x"))[1] == 0.2


# --- one document, one key -------------------------------------------------


def test_arxiv_abs_pdf_and_versions_are_one_document():
    assert (
        canonical_url("https://arxiv.org/abs/2410.08315")
        == canonical_url("http://arxiv.org/pdf/2410.08315v2")
        == canonical_url("https://www.arxiv.org/abs/2410.08315v1/")
    )


def test_tracking_parameters_and_trailing_slashes_change_nothing():
    assert canonical_url("https://a.org/x/?utm_source=g&id=3") == canonical_url("https://a.org/x?id=3")
    assert canonical_url("https://a.org/x?id=3") != canonical_url("https://a.org/x?id=4")


def test_mirror_titles_match_without_their_decoration():
    paper = "Wavefront shaping in multimode fibers by transmission matrix engineering"
    assert title_key(f"{paper} - arXiv") == title_key(f"[PDF] {paper}")
    assert title_key(f"(PDF) {paper}") == title_key(paper)


# --- scoring ---------------------------------------------------------------


def test_relevance_counts_the_title_above_the_snippet():
    terms = anchor_sequence("fine-tune a large language model with LoRA")
    in_title = relevance(r("https://a.org", title="LoRA for large language models"), terms)
    in_snippet = relevance(r("https://a.org", snippet="LoRA for large language models"), terms)
    assert in_title > in_snippet > 0


def test_a_title_naming_the_subject_differently_still_counts():
    """"Finetuning LLMs" is "fine-tune a large language model"; matching words
    exactly scored this page as barely relevant and it was never read."""
    terms = anchor_sequence("How to fine-tune a large language model with LoRA")
    page = r("https://magazine.sebastianraschka.com/p/lora",
             title="Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)")
    assert relevance(page, terms) == 1.0


def test_an_acronym_in_the_question_matches_a_title_that_spells_it_out():
    terms = anchor_sequence("LLM quantization")
    page = r("https://a.org", title="Large language model quantization in practice")
    assert relevance(page, terms) == 1.0


def test_an_on_topic_blog_beats_a_paper_that_only_grazes_the_subject():
    """The LoRA case: scholarliness must not outweigh relevance."""
    question = "How to fine-tune a large language model with LoRA"
    picked = select_sources(
        [[
            r("https://arxiv.org/abs/2401.00001", title="Benchmarking adapters for financial tabular data"),
            r("https://magazine.sebastianraschka.com/p/lora",
              title="Practical tips for fine-tuning large language models using LoRA"),
        ]],
        question,
        limit=1,
    )
    assert "sebastianraschka" in picked.chosen[0].result.url


def test_among_on_topic_pages_the_scholarly_one_leads():
    """The entropy case: Reddit ranked first, Wikipedia second."""
    picked = select_sources(
        [[
            r("https://www.reddit.com/r/Physics/entropy", title="Can somebody explain what entropy is?"),
            r("https://en.wikipedia.org/wiki/Entropy", title="Entropy - Wikipedia"),
            r("https://www.sciencedirect.com/topics/entropy", title="Entropy - an overview"),
        ]],
        "What is entropy?",
        limit=3,
    )
    urls = [c.result.url for c in picked.chosen]
    assert urls.index("https://www.reddit.com/r/Physics/entropy") == 2


def test_a_low_priority_page_is_still_read_when_nothing_better_was_found():
    picked = select_sources(
        [[r("https://www.reddit.com/r/x", title="entropy"), r("https://arxiv.org/abs/1", title="entropy")]],
        "entropy",
        limit=12,
    )
    assert len(picked.chosen) == 2


def test_a_page_found_by_several_queries_counts_once_and_ranks_higher():
    shared = r("https://zohaib.me/lora", title="LoRA guide")
    picked = select_sources(
        [[r("https://a.example/lora", title="LoRA guide"), shared],
         [r("https://b.example/lora", title="LoRA guide"), shared]],
        "LoRA guide",
        limit=12,
    )
    urls = [c.result.url for c in picked.chosen]
    assert urls.count("https://zohaib.me/lora") == 1
    assert urls[0] == "https://zohaib.me/lora"


def test_a_terse_title_found_by_most_queries_is_still_read():
    """Hugging Face's LoRA page: one word of the question in its title, but
    three of four on-topic queries returned it."""
    docs = r("https://huggingface.co/docs/peft/lora", title="LoRA (Low-Rank Adaptation)")
    filler = [
        r(f"https://site{q}{i}.example/x", title="Fine-tuning large language models with LoRA")
        for q in range(4) for i in range(3)
    ]
    per_query = [[docs, *filler[0:3]], [docs, *filler[3:6]], [docs, *filler[6:9]], filler[9:12]]
    picked = select_sources(per_query, "How to fine-tune a large language model with LoRA", limit=4)
    assert docs.url in [c.result.url for c in picked.chosen]


# --- spread ----------------------------------------------------------------


def test_one_scholarly_host_does_not_take_every_slot():
    papers = [r(f"https://arxiv.org/abs/2401.{i:05d}", title=f"Diffusion paper number {i}") for i in range(10)]
    others = [r(f"https://site{i}.example/diffusion", title=f"Diffusion notes {i}") for i in range(5)]
    picked = select_sources([papers + others], "diffusion", limit=8)
    hosts = [c.group for c in picked.chosen]
    assert 1 < hosts.count("arxiv.org") < 8


def test_the_same_paper_on_another_host_waits_behind_distinct_pages():
    paper = "Wavefront shaping in multimode fibers by transmission matrix engineering"
    picked = select_sources(
        [[
            r("https://arxiv.org/abs/2001.00001", title=f"[2001.00001] {paper}"),
            r("https://pubs.aip.org/app/article/5/1", title=paper),
            r("https://www.semanticscholar.org/paper/1", title=f"[PDF] {paper}"),
            r("https://opg.optica.org/oe/1", title="Online learning of the transmission matrix of dynamic scattering media"),
        ]],
        "transmission matrix engineering",
        limit=2,
    )
    urls = [c.result.url for c in picked.chosen]
    assert "https://opg.optica.org/oe/1" in urls
    assert picked.mirrors_set_aside == 2


def test_every_query_gets_a_page_before_any_query_gets_two():
    """What the round-robin guaranteed, kept: the queries aimed elsewhere are
    not starved by the one closest to the question."""
    per_query = [
        [r(f"https://arxiv.org/abs/2401.0000{i}", title=f"entropy paper {i}") for i in range(5)],
        [r("https://www.reddit.com/r/a", title="something else")],
    ]
    picked = select_sources(per_query, "entropy", limit=2)
    assert "https://www.reddit.com/r/a" in [c.result.url for c in picked.chosen]


def test_nothing_found_selects_nothing():
    picked = select_sources([[], []], "q", limit=12)
    assert picked.chosen == [] and picked.considered == 0
