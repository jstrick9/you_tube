import re

from autotube.scriptwriter import (
    _attach_missing_evidence,
    _source_evidence_quote,
    has_specific_visual_subject,
    visual_subject_for_topic,
)


SOURCE = (
    "Archaeologists found a stone sphere in the valley. "
    "The stone sphere measures two meters across and weighs several tons. "
    "It was carved from granite by an unknown group."
)


def _words(text):
    return re.findall(r"[\w]+(?:['’][\w]+)?", text)


def test_visual_subject_resolves_nonvisual_modifiers_to_concrete_subjects():
    grounded_city = {"title": "Derinkuyu underground city"}
    assert visual_subject_for_topic(
        {"script_subject": "Ancient Mysteries Unearthed"}, grounded_city,
    ) == "Derinkuyu underground city"
    assert visual_subject_for_topic(
        {"script_subject": "Lost 1,000‑Year‑Old City"}, grounded_city,
    ) == "Derinkuyu underground city"
    assert visual_subject_for_topic(
        {"script_subject": "Cleopatra eyeliner poison"}, {"title": "Kohl"},
    ) == "Cleopatra eyeliner"
    assert visual_subject_for_topic(
        {"script_subject": "Mountain Goats Jumps"}, {"title": "Goats"},
    ) == "Mountain Goats"
    assert visual_subject_for_topic(
        {"script_subject": "Terrifying Carnivorous Plants"}, {"title": "Carnivorous plant"},
    ) == "Carnivorous Plants"
    assert visual_subject_for_topic(
        {"script_subject": "The Evolution of Cycles"}, {"title": "Bicycle"},
    ) == "Bicycle"
    assert visual_subject_for_topic(
        {"script_subject": "Giant Stone Spheres"}, {"title": "Stone spheres"},
    ) == "Stone Spheres"
    # Trend framing such as "last goodbye" is not the physical subject to depict.
    assert visual_subject_for_topic(
        {"script_subject": "Voyager 1's Last Goodbye"}, {"title": "Voyager 1"},
    ) == "Voyager"
    assert visual_subject_for_topic(
        {"script_subject": "Green Light on Jupiter"}, {"title": "Juno (spacecraft)"},
    ) == "Jupiter"
    assert visual_subject_for_topic(
        {"script_subject": "Jeep Missile Launcher"}, {"title": "Nuclear command and control"},
    ) == "Jeep"
    # Abstract environmental framing can resolve to the specific grounded object/place;
    # the source-angle gate still decides whether that source supports the live claim.
    assert visual_subject_for_topic(
        {"script_subject": "Deep Sea Discovery"}, {"title": "Hydrothermal vent"},
    ) == "Hydrothermal vent"
    assert visual_subject_for_topic(
        {"script_subject": "Deep Ocean Pressure"}, {"title": "Mariana Trench"},
    ) == "Mariana Trench"


def test_abstract_deep_sea_exploration_is_not_mistaken_for_a_visual_subject():
    abstract = visual_subject_for_topic(
        {"script_subject": "Deep-sea exploration"}, {"title": "Deep-sea exploration"},
    )
    assert abstract == "Deep-sea exploration"
    assert not has_specific_visual_subject(abstract)
    assert has_specific_visual_subject("Hydrothermal vent")
    assert has_specific_visual_subject("Stone Sphere")


def test_action_heavy_car_trend_normalizes_to_a_physical_subject():
    assert visual_subject_for_topic(
        {"script_subject": "Car AI calling driver sleepy"},
        {"title": "Driver drowsiness detection"},
    ) == "Car driver"
    assert has_specific_visual_subject("Car driver")


def test_produce_skips_abstract_visual_subject_before_any_script_draft(monkeypatch):
    import yaml
    from pathlib import Path
    from autotube import scriptwriter

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    source = {"title": "Deep-sea exploration", "text": "Deep-sea exploration describes a broad activity. " * 80}

    class NoDraftLLM:
        lite = False
        calls = 0

        def json(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("an abstract subject should be rejected before drafting")

    llm = NoDraftLLM()
    monkeypatch.setattr(scriptwriter, "ground", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(scriptwriter, "subject_is_depictable", lambda *_args, **_kwargs: True)
    writer = scriptwriter.ScriptWriter(cfg, llm, object())
    topic = {
        "topic": "A Shocking Discovery in the Deep Sea",
        "script_subject": "Deep-sea exploration",
        "wiki_query": "Deep-sea exploration",
        "angle": "deep-sea exploration describes a broad activity",
    }

    assert writer.produce(topic, {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "test"}) is None
    assert llm.calls == 0


def test_missing_evidence_is_backfilled_only_with_a_source_matched_quote():
    segments = [
        {"text": "This strange sphere was found in a field."},
        {"text": "The stone sphere is two meters wide and weighs several tons."},
        {"text": "The temple vanished when its emperor disappeared."},
    ]

    attached = _attach_missing_evidence(segments, SOURCE, min_words=5, max_words=9)

    assert attached == 1
    assert segments[0]["evidence"] == ""  # the hook is allowed to remain empty
    quote = segments[1]["evidence"]
    assert quote and quote in SOURCE
    assert 5 <= len(_words(quote)) <= 9
    assert segments[2]["evidence"] == ""  # topical mismatch is not evidence


def test_evidence_backfill_rejects_unsupported_numbers_and_reused_quotes():
    unsupported = "The stone sphere measures 17 meters across and weighs several tons."
    assert _source_evidence_quote(unsupported, SOURCE, min_words=5, max_words=9) is None

    segments = [
        {"text": "The hook names the stone sphere."},
        {"text": "The stone sphere measures two meters across and weighs several tons."},
        {"text": "The stone sphere measures two meters across and weighs several tons."},
    ]

    attached = _attach_missing_evidence(segments, SOURCE, min_words=5, max_words=9)

    assert attached == 1
    assert segments[0]["evidence"] == ""
    assert segments[1]["evidence"] in SOURCE
    assert segments[2]["evidence"] == ""  # the same source passage is not counted twice


def test_real_writer_validator_backfills_an_omitted_evidence_field():
    import pytest
    import yaml
    from pathlib import Path
    from autotube.scriptwriter import ScriptWriter

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    captured = {}

    class FakeLLM:
        lite = False
        last_used = "fake:model"

        def json(self, _system, _user, validate=None, **_kwargs):
            captured["validate"] = validate
            captured["user"] = _user
            raise RuntimeError("capture the production validator")

    class FakeStrategy:
        pass

    writer = ScriptWriter(cfg, FakeLLM(), FakeStrategy())
    try:
        writer.write(
            {"topic": "The Stone Sphere", "script_subject": "The Stone Sphere",
             "category": "history_mystery", "angle": ""},
            {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "test"},
            {"title": "Stone sphere", "url": "https://example.invalid", "text": SOURCE,
             "summary": "A sphere made from stone."},
        )
    except RuntimeError:
        pass

    script = {"title": "The Stone Sphere mystery", "segments": [
        {"text": "The Stone Sphere shocked everyone.", "visual": {"shows": "a dark cave", "queries": ["dark cave", "cave wall"]}},
        {"text": "The stone sphere is two meters wide and weighs several tons.",
         "visual": {"shows": "a dark cave", "queries": ["dark cave", "cave wall"]}},
        {"text": "The temple vanished when its emperor disappeared.",
         "visual": {"shows": "a dark cave", "queries": ["dark cave", "cave wall"]}},
        {"text": "The stone sphere remains a mystery today.",
         "visual": {"shows": "a dark cave", "queries": ["dark cave", "cave wall"]}},
    ]}
    with pytest.raises(AssertionError, match="no shot shows"):
        captured["validate"](script)

    assert "CONCRETE SUBJECT TO NAME AND SHOW: Stone Sphere" in captured["user"]
    assert "Write the script as exactly 4 or 5 segments. Prefer exactly FIVE by default" in captured["user"]
    assert "FINAL SHAPE CHECK" in captured["user"]
    assert "A 'How/Why' title must explain how/why" in captured["user"]
    assert script["segments"][0]["evidence"] == ""
    assert script["segments"][1]["evidence"] in SOURCE
    assert script["segments"][2]["evidence"] == ""
