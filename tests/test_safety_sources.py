"""Context-aware topic safety (§3.4) and the widened grounding corpora (§3.3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import safety, sources  # noqa: E402
from autotube.common import load_config  # noqa: E402

CFG = load_config()


# ── §3.4 safety triage ────────────────────────────────────────────────────────
def test_safe_subjects_the_old_blocklist_deleted_are_now_reviewable():
    """`war`, `crash`, `leak`, `trial`, `cure`, `flood`... appear constantly in safe content.
    They must no longer be an automatic rejection."""
    for subject in ["Crash test dummy", "Thirty Years' War", "Vacuum leak in the ISS airlock",
                    "Clinical trial of penicillin", "Flood basalt of the Deccan Traps",
                    "The 1906 San Francisco earthquake", "Trial of the Pyx", "Cure for scurvy",
                    "Hurricane hunters aircraft", "Tornado Alley formation"]:
        verdict, _ = safety.screen(subject, CFG)
        assert verdict == "review", f"{subject!r} should go to context review, got {verdict}"


def test_hard_blocked_subjects_never_reach_a_classifier():
    for subject in ["Serial killer documentary", "Graphic torture methods", "OnlyFans leak",
                    "School shooting anniversary", "Famous massacre sites", "beheading video"]:
        verdict, term = safety.screen(subject, CFG)
        assert verdict == "block" and term, subject


def test_ordinary_subjects_cost_nothing():
    for subject in ["Olympus Mons", "How vaccines were invented", "The deepest lake on Earth",
                    "Why flamingos are pink"]:
        assert safety.screen(subject, CFG) == ("ok", None), subject


def test_titles_are_stricter_than_narration():
    assert safety.screen_title("The Murder That Changed Forensics", CFG) == "murder"
    assert safety.screen_title("The Trial of the Pyx: 800 Years of Testing Coins", CFG) is None
    assert safety.screen_title("How Crash Tests Got Safer", CFG) is None


def test_classifier_decides_sensitive_subjects():
    class Yes:
        def json(self, *a, **k):
            return {"allowed": True, "reason": "non-graphic engineering history",
                    "angle": "the evolution of vehicle safety testing"}

    class No:
        def json(self, *a, **k):
            return {"allowed": False, "reason": "real victims of a recent event", "angle": ""}

    ok, _ = safety.check_subject("Crash test dummy", "history of vehicle safety testing", CFG, Yes())
    assert ok
    bad, reason = safety.check_subject("Crash test dummy", "a fatal motorway pileup", CFG, No())
    assert not bad and "sensitive subject" in reason


def test_classifier_outage_falls_back_to_rejecting():
    """Never less safe than the keyword list it replaces."""
    class Down:
        def json(self, *a, **k):
            raise RuntimeError("all providers down")

    ok, reason = safety.check_subject("Thirty Years' War", "a date reference", CFG, Down())
    assert not ok and "classifier unavailable" in reason
    assert not safety.check_subject("Thirty Years' War", "x", CFG, None)[0]


def test_malformed_safety_term_lists_fall_back_to_safe_defaults():
    cfg = {"safety": {"hard_blocked": "murder", "title_blocked": "xxx", "sensitive": "war"}}
    assert safety.screen("serial killer documentary", cfg)[0] == "block"
    assert safety.screen("Thirty Years' War", cfg)[0] == "review"
    assert safety.screen_title("A murder mystery", cfg) == "murder"


def test_malformed_context_review_setting_blocks_sensitive_topics():
    cfg = {"safety": {"context_review": "false"}}
    assert safety.screen("Thirty Years' War", cfg)[0] == "block"


def test_legacy_blocked_topics_config_still_works():
    legacy = {"compliance": {"blocked_topics": ["war", "crash"]}}
    assert safety.screen("Thirty Years' War", legacy)[0] == "review"
    assert safety.screen("Olympus Mons", legacy) == ("ok", None)


def test_check_script_blocks_hard_terms_and_bad_titles():
    issues = safety.check_script("The Murder Weapon", "a perfectly ordinary line", CFG)
    assert any("never allowed in a title" in i for i in issues)
    assert safety.check_script("A Fine Title", "it was pure torture to build", CFG)
    assert safety.check_script("How Crash Tests Got Safer", "the sled hits the barrier at speed", CFG) == []


# ── §3.3 grounding corpora ────────────────────────────────────────────────────
def test_relevance_requires_the_subject_to_be_the_topic():
    """The sandhill-crane trap: a passage that merely name-drops the subject is not evidence."""
    sandhill = ("A pair of sandhill cranes search for food. Florida sandhill cranes stay with the same "
                "mate for several years. Like their endangered relatives the whooping cranes, sandhills "
                "live to be older than most birds. Some sandhill cranes live up to 20 years.")
    assert not sources._relevant("KSC-07pd3638", sandhill, "Whooping crane")
    assert sources._relevant("Whooping crane recovery", sandhill, "Whooping crane")  # named in title
    real = ("The whooping crane is the tallest North American bird. Whooping cranes were reduced to 21 "
            "individuals before recovery efforts began.")
    assert sources._relevant("Grus americana", real, "Whooping crane")


def test_multiword_subjects_match_as_a_phrase():
    assert sources._mentions("the crane whooped loudly", "Whooping crane") == []
    assert sources._mentions("a whooping crane landed", "Whooping crane")
    assert sources._mentions("whooping cranes migrate", "Whooping crane")


def test_wikidata_time_rejects_deep_time_and_nonsense():
    # Olympus Mons inception is '-3830000000-00-00T00:00:00Z' precision 2 (billions of years).
    # Reading the leading digits gives 'the year 3830' — confidently absurd.
    assert sources._year({"time": "-3830000000-00-00T00:00:00Z", "precision": 2}) is None
    assert sources._year({"time": "+1889-00-00T00:00:00Z", "precision": 9}) == "1889"
    assert sources._year({"time": "-0044-00-00T00:00:00Z", "precision": 9}) == "44 BC"
    assert sources._year({"time": "+1800-00-00T00:00:00Z", "precision": 7}) is None   # century


def test_enrich_is_fail_soft_and_labels_everything(monkeypatch):
    monkeypatch.setitem(sources.FETCHERS, "boom", lambda s, limit=5: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setitem(sources.FETCHERS, "good", lambda s, limit=5: [
        {"kind": "good", "title": "T", "url": "http://u", "text": "x" * 300, "publisher": "Pub"}])
    # an unknown or exploding corpus must not break grounding
    try:
        block, credits = sources.enrich("Subject", ["good", "nope"])
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"enrich must fail soft, raised {e}")
    assert block.startswith('From Pub ("T"):') and credits[0]["url"] == "http://u"
    assert sources.enrich("Subject", []) == ("", [])


def test_enrich_respects_the_character_cap(monkeypatch):
    monkeypatch.setitem(sources.FETCHERS, "big", lambda s, limit=5: [
        {"kind": "big", "title": f"T{i}", "url": "u", "text": "y" * 1400, "publisher": "P"} for i in range(5)])
    block, _ = sources.enrich("Subject", ["big"], max_chars=1500)
    assert len(block) <= 1600
