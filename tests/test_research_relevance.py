from autotube import research
from autotube.scriptwriter import source_supports_angle


def _article(text):
    return {"query": {"pages": {"1": {"extract": text}}}}


def test_article_text_keeps_deep_relevant_source_paragraphs_and_the_lead(monkeypatch):
    lead = "Gold is a chemical element with symbol Au and atomic number 79. " + (
        "The article lead adds brief context that is not the cosmic-origin claim. " * 10)
    article = (
        lead + "\n\n\n"
        "Uses\n\n\n" + ("Gold is used in jewelry and electronics. " * 500) + "\n\n\n"
        "Origin\n\n\nGold production in the universe\n\n\n"
        "Gold in the universe is produced through cosmic processes. Three sources for gold formation are "
        "supernova nucleosynthesis, neutron star collisions, and magnetar flares.\n\n\n"
        "All three sources involve the r-process, which forms elements heavier than iron."
    )
    monkeypatch.setattr(research, "get_json", lambda *_args, **_kwargs: _article(article))

    excerpt = research.article_text(
        "Gold", max_chars=1800,
        focus_query="Where does gold come from? Gold formed through neutron star collisions.",
    )

    assert excerpt.startswith("Gold is a chemical element")
    assert "not the cosmic-origin claim" in excerpt
    assert "neutron star collisions" in excerpt
    assert "r-process" in excerpt
    assert source_supports_angle(
        "Gold can form in neutron-star collisions and magnetar flares, alongside supernovae.",
        "Gold", excerpt,
    )
    assert len(excerpt) <= 1800
    assert "jewelry and electronics" not in excerpt


def test_article_focus_matches_accents_and_british_spelling_for_retrieval_only(monkeypatch):
    article = (
        "The koala is an Australian marsupial.\n\n\n"
        "Reproduction and development\n\n\n"
        "A joey eats a faecal pap from its mother. The pap is more liquid than regular faeces and is filled with bacteria."
    )
    monkeypatch.setattr(research, "get_json", lambda *_args, **_kwargs: _article(article))

    excerpt = research.article_text(
        "Koala", max_chars=1200,
        focus_query="Por qué los bebés koala comen la popó de su madre",
    )

    assert "The koala is an Australian marsupial" in excerpt
    assert "faecal pap" in excerpt
    assert "faeces" in excerpt


def test_focused_article_retrieval_falls_back_to_the_lead_when_no_terms_match(monkeypatch):
    article = "The topic is described in this short lead.\n\n\nOther section\n\n\nA separate fact."
    monkeypatch.setattr(research, "get_json", lambda *_args, **_kwargs: _article(article))

    excerpt = research.article_text("Topic", max_chars=200, focus_query="no matching words")

    assert excerpt.startswith("The topic is described in this short lead.")
    assert "Other section" in excerpt and "A separate fact." in excerpt
    assert len(excerpt) <= 200
