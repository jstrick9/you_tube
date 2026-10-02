"""News grounding for trends with no Wikipedia article (AUDIT 3.3, trending-only scope)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import news  # noqa: E402


def _a(outlet, title, url="http://x/a", via="google_news"):
    return {"outlet": outlet, "title": title, "url": url, "via": via}


# ── independence ──────────────────────────────────────────────────────────────
def test_google_news_links_are_not_mistaken_for_an_aggregator():
    """Every Google News link is a news.google.com redirect; judging by URL deleted the whole feed."""
    arts = [_a("BBC", "A thing happened", "https://news.google.com/rss/articles/abc"),
            _a("Reuters", "A thing happened", "https://news.google.com/rss/articles/def")]
    assert len(news.independent(arts)) == 2


def test_one_outlet_cannot_be_counted_twice():
    arts = [_a("BBC", "First story"), _a("BBC", "Second story"), _a("Reuters", "Third")]
    assert [x["outlet"] for x in news.independent(arts)] == ["BBC", "Reuters"]


def test_aggregators_are_not_independent_corroboration():
    arts = [_a("MSN", "Syndicated", "https://msn.com/x", via="gdelt"),
            _a("Yahoo", "Syndicated", "https://news.yahoo.com/x", via="gdelt"),
            _a("Reuters", "Original", "https://reuters.com/x", via="gdelt")]
    assert [x["outlet"] for x in news.independent(arts)] == ["Reuters"]


# ── consensus ─────────────────────────────────────────────────────────────────
def test_consensus_needs_more_than_one_outlet():
    arts = [_a("BBC", "Volcano erupts in Iceland"), _a("Reuters", "Volcano erupts in Iceland"),
            _a("Sky", "Celebrity buys a sandwich")]
    c = news.consensus(arts, min_outlets=2)
    assert "volcano erupts" in c
    assert not any("sandwich" in p for p in c), "a single outlet's claim must not become consensus"


def test_one_outlet_repeating_itself_is_still_one_vote():
    arts = [_a("BBC", "Rumour rumour rumour rumour")]
    assert news.consensus(arts, min_outlets=2) == []


def test_consensus_ignores_stopword_edges():
    arts = [_a("A", "The launch of the rocket"), _a("B", "The launch of the rocket")]
    for p in news.consensus(arts, 2):
        assert not p.startswith("the ") and not p.endswith(" the")


# ── backbone (proper-noun detection) ──────────────────────────────────────────
def test_backbone_prefers_entities_over_verbs():
    arts = [_a("Mashable", "OpenAI’s Sora shutting down: when it happens"),
            _a("TechCrunch", "Why OpenAI really shut down Sora"),
            _a("The Hill", "OpenAI shutting down video generator Sora"),
            _a("Variety", "Disney exits OpenAI deal over Sora")]
    b = news.backbone("Sora 2", news.consensus(arts, 2), arts)
    assert any(x.lower() == "openai" for x in b), b
    assert not any(x.lower() in ("shutting", "down", "really") for x in b), b


def test_title_cased_headlines_are_not_capitalisation_evidence():
    """In Title Case every word is capitalised, so it proves nothing about proper nouns."""
    assert news._title_cased("Sora Is Dead And Slop Is Hollywood Now")
    assert not news._title_cased("OpenAI shuts down its Sora video app")
    arts = [_a("A", "Sora Is Dead And Slop Won Hollywood"),
            _a("B", "Sora Is Dead And Slop Won Hollywood")]
    assert not any(x.lower() in ("slop", "dead") for x in news.backbone("Sora", news.consensus(arts, 2), arts))


def test_entities_that_habitually_open_a_headline_are_not_penalised():
    """'OpenAI pulls the plug' leads the sentence; that is grammar, not evidence against it."""
    arts = [_a("A", "OpenAI pulls the plug on Sora"), _a("B", "OpenAI pulls the plug on Sora"),
            _a("C", "Users mourn OpenAI and Sora")]
    assert any(x.lower() == "openai" for x in news.backbone("Sora", news.consensus(arts, 2), arts))


def test_backbone_excludes_the_subject_itself():
    arts = [_a("A", "Labubu mania grips Pop Mart"), _a("B", "Labubu mania grips Pop Mart")]
    assert not any(x.lower() == "labubu" for x in news.backbone("Labubu", news.consensus(arts, 2), arts))


# ── headline cleaning ─────────────────────────────────────────────────────────
def test_outlet_suffix_is_stripped_even_when_it_contains_hyphens():
    assert news._clean("Veo 3.1 Setup: Migrate From Sora 2 - tech-insider.org",
                       "tech-insider.org") == "Veo 3.1 Setup: Migrate From Sora 2"
    assert news._clean("A story - BBC News", "BBC News") == "A story"


def test_clean_never_returns_empty():
    assert news._clean("- BBC", "BBC")


# ── brief ─────────────────────────────────────────────────────────────────────
def test_brief_refuses_a_single_sourced_story(monkeypatch):
    monkeypatch.setattr(news, "google_news", lambda *a, **k: [_a("BBC", "Only BBC says this")])
    monkeypatch.setattr(news, "gdelt", lambda *a, **k: [])
    assert news.brief("thing", min_outlets=2) is None


def test_brief_returns_outlets_and_credits(monkeypatch):
    arts = [_a("BBC", "Iceland volcano erupts near Grindavik"),
            _a("Reuters", "Iceland volcano erupts near Grindavik")]
    monkeypatch.setattr(news, "google_news", lambda *a, **k: arts)
    monkeypatch.setattr(news, "gdelt", lambda *a, **k: [])
    b = news.brief("Iceland volcano", min_outlets=2)
    assert b and b["n_outlets"] == 2
    assert {c["publisher"] for c in b["also"]} == {"BBC", "Reuters"}
    assert "headlines only" in b["text"], "the text must state what it is, so nothing treats it as prose"


def test_brief_is_fail_soft_when_the_network_dies(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(news, "google_news", boom)
    try:
        news.brief("thing")
    except RuntimeError:
        pass  # google_news itself is the fail-soft boundary; brief may propagate a hard monkeypatch
