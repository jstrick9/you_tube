from autotube import common, research, scriptwriter

DOG_LAPPING_QUERY = "Dogs lap using acceleration-driven open pumping"
DOG_LAPPING_ANGLE = "Dogs curl their tongues to pull a larger liquid column into their mouths."
DOG_LAPPING_ABSTRACT = (
    "Dogs lap because they have incomplete cheeks and cannot suck. When lapping, a dog's tongue pulls a liquid "
    "column from the bath. We measured lapping in 19 dogs and used the results to generate a physical model of "
    "the tongue's interaction with the air-fluid interface. The results demonstrate that effects of acceleration "
    "govern lapping frequency, which suggests that dogs curl the tongue to create a larger liquid column."
)


def test_curated_lapping_angle_is_supported_by_the_primary_study():
    assert scriptwriter.source_supports_angle(DOG_LAPPING_ANGLE, "Dog", DOG_LAPPING_ABSTRACT)


def test_ground_prioritizes_exact_pubmed_query_and_spends_only_the_remaining_budget(monkeypatch):
    article = "Wikipedia's general dog article. " * 300  # deliberately longer than the enrichment cutoff
    primary = 'From PNAS ("Dogs lap using acceleration-driven open pumping"): ' + "study details " * 30
    calls = []

    monkeypatch.setattr(research, "narrow_title", lambda *_args, **_kwargs: "Dog")
    monkeypatch.setattr(research, "wiki_summary", lambda *_args, **_kwargs: {
        "title": "Dog", "description": "domesticated mammal", "extract": "General dog background."})
    monkeypatch.setattr(research.safety, "check_subject", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(research, "is_living_person", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(research, "article_text", lambda *_args, **_kwargs: article)
    monkeypatch.setattr(research, "article_images", lambda *_args, **_kwargs: [])

    def fake_enrich(subject, enabled, max_chars):
        calls.append((subject, list(enabled), max_chars))
        if enabled == ["pubmed"]:
            return primary, [{"title": "PNAS — dog lapping study", "url": "https://doi.org/10.1073/pnas.1514842112",
                              "publisher": "PNAS"}]
        return "Other configured research.", []

    monkeypatch.setattr(research.sources, "enrich", fake_enrich)
    cfg = {
        "content": {"allow_living_people": False},
        "compliance": {"blocked_topics": []},
        "research": {"extra_sources": ["wikidata", "nasa", "arxiv", "pubmed"],
                     "enrich_below_chars": 7000, "enrich_max_chars": 4000},
    }

    source = research.ground("Dog Paani Kaise Peeta Hai", "Dog", cfg=cfg, research_query=DOG_LAPPING_QUERY)

    assert source is not None
    assert calls[0][0:2] == (DOG_LAPPING_QUERY, ["pubmed"])
    assert calls[1][0] == "Dog"
    assert "pubmed" not in calls[1][1]  # never spend the remaining budget on a broad PubMed search
    assert calls[1][2] < calls[0][2]
    assert source["text"].startswith(primary)
    assert source["text"][:2600].startswith(primary)  # even the lite model's excerpt sees the primary abstract
    assert "acceleration-driven open pumping" in source["text"] or "study details" in source["text"]
    assert any("PNAS" in credit["title"] for credit in source["also"])


def test_script_production_forwards_its_curated_research_query_to_grounding(monkeypatch):
    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = common.load_config()
    writer.lang = "en"
    writer.llm = object()
    captured = {}
    source = {"title": "Dog", "text": "Background about dogs, not the study-specific angle."}

    def fake_ground(*_args, **kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr(scriptwriter, "ground", fake_ground)
    monkeypatch.setattr(writer, "_repair_trend_angle", lambda *_args, **_kwargs: None)
    topic = {
        "topic": "Dog Paani Kaise Peeta Hai? Tongue Ka Asli Secret!",
        "script_subject": "Dog", "wiki_query": "Dog", "angle": DOG_LAPPING_ANGLE,
        "research_query": DOG_LAPPING_QUERY,
    }

    # The intentionally unhelpful source fails the regular source-angle gate before any script draft.
    assert writer.produce(topic, {}) is None
    assert captured["research_query"] == DOG_LAPPING_QUERY
