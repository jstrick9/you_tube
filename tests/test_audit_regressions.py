"""Regressions found by the post-implementation audit.

Each of these is a defect that the unit tests missed because they verified the
components in isolation and never checked that they were wired together correctly.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from autotube import common, originality, provenance, series  # noqa: E402
from autotube.scriptwriter import ScriptWriter  # noqa: E402

CFG = common.load_config()


class _Rev:
    lite = False

    def json(self, *a, **k):
        return {"score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8,
                "loops": True, "factual_errors": [], "misleading_title": False,
                "advertiser_friendly": True, "policy_concerns": [],
                "value_add": "A concise explanation of the source-backed mystery.", "fixes": []}


def _writer(history=None):
    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = _Rev(), 4000
    w.cfg = {**CFG, "content": dict(CFG["content"], require_grounding=False)}
    w._history = list(history or [])
    return w


SRC = {"title": "S", "text": "keepers vanished lighthouse coats door bolted "* 40}


def _script(title, aside="Tidy of them."):
    return {"title": title, "segments": [
        {"text": "Three lighthouse keepers vanished from a locked station"},
        {"text": "Two coats were missing, but one still hung beside the door", "aside": aside},
        {"text": "The logbook gave no warning or reason to explain their absence"},
        {"text": "so the locked door never explained their disappearance"}]}


# ── Defect 1: the similarity gate was blind inside a single run ─────────────
def test_two_near_identical_scripts_in_one_run_are_caught():
    """history.json is only written at the END of a run, and a run makes up to three
    videos. Same-run videos are the most likely to collide: same day's trends, same
    lane, same format pool. Before the fix both of these were approved."""
    w = _writer()
    assert w.check(_script("T1"), SRC)[0] is True
    ok, review = w.check(_script("T2", aside="Tidy indeed."), SRC)
    assert ok is False
    assert any("too close to a published episode" in i for i in review["issues"])


def test_a_genuinely_different_second_video_in_the_same_run_still_passes():
    """The fix must not make a run of three videos impossible."""
    w = _writer()
    assert w.check(_script("T1"), SRC)[0] is True
    other = {"title": "T2", "segments": [
        {"text": "Roman concrete can heal its own cracks when seawater gets in"},
        {"text": "Saltwater feeds crystals that grow through the damaged stone", "aside": "Show off."},
        {"text": "They fill the gap until the seawall is sealed again"},
        {"text": "so the break becomes part of the repair"}]}
    assert w.check(other, {"title": "S2", "text": "roman concrete seawater crystals " * 40})[0] is True


def test_a_rejected_draft_is_not_held_against_its_own_rewrite():
    """A rejected draft is about to be rewritten on the same topic. Registering it
    would guarantee the retry collides with it and the topic would be abandoned."""
    w = _writer()
    bad = _script("T1", aside="")          # no aside -> rejected for having no voice
    assert w.check(bad, SRC)[0] is False
    assert w.check(_script("T1"), SRC)[0] is True, "the rewrite must not collide with the draft it replaces"


def test_production_only_checks_a_script_once_per_draft():
    """The in-flight registration is only safe because an approved script is never
    re-checked; produce() returns as soon as check() approves."""
    import inspect
    src = inspect.getsource(ScriptWriter.produce)
    assert src.count("self.check(") == 1
    assert "if ok:" in src and "return script, source, review" in src


# ── Defect 2: appeal evidence was written for unpublished videos ────────────
def test_provenance_is_only_recorded_for_videos_that_actually_published(tmp_path, monkeypatch):
    """An upload that raised leaves a rendered file that is not on the channel. A
    dossier claiming it would be inflated evidence, and a reviewer who checks one
    claimed episode and cannot find it discredits the whole archive."""
    import inspect

    from autotube import pipeline
    src = inspect.getsource(pipeline.run)
    assert 'if entry.get("video_id"):' in src, "record must be gated on a real upload"
    i_guard = src.index('if entry.get("video_id"):')
    i_call = src.index("provenance.record(")
    assert i_guard < i_call, "the guard must precede the call"


def test_dossier_does_not_count_an_episode_that_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "DIR", tmp_path / "p")
    res = {"script": {"segments": [{"text": "a", "evidence": "b", "aside": "c"}]},
           "source": {"title": "S", "url": "https://x/y"}, "topic": {}, "plan": {},
           "review": {"independent": True}}
    provenance.record(res, {"title": "published", "video_id": "abc", "created_at": "2026-10-03"}, CFG)
    assert provenance.dossier()["episodes"] == 1


# ── Defect 3: series.remit was config that nothing read ─────────────────────
def test_every_configured_series_remit_reaches_the_writer_prompt():
    import inspect
    src = inspect.getsource(ScriptWriter.write)
    assert "series_rule" in src, "remit must be built"
    assert "{person_rule}{series_rule}" in src, "remit must be interpolated into the prompt"
    assert '_spec["remit"]' in src


def test_no_config_key_under_series_is_unread():
    """The audit found remit defined and never used. Keep it that way."""
    body = "\n".join(p.read_text() for p in Path(__file__).resolve().parent.parent
                     .joinpath("autotube").glob("*.py"))
    for key, spec in CFG["content"]["series"].items():
        for field in spec:
            assert f'"{field}"' in body or f"'{field}'" in body, f"series.{key}.{field} is never read"


@pytest.mark.parametrize("cat,tag", [("mysteries", "CASE"), ("space", "FIELD")])
def test_remit_is_defined_for_every_live_category(cat, tag):
    key = series.assign(cat, CFG)
    spec = series.definitions(CFG)[key]
    assert spec["tag"] == tag
    assert len(spec.get("remit", "")) > 20, "a remit too vague to constrain anything is not a remit"


# ── Previously untested: subscriber trajectory ──────────────────────────────
def test_subs_progress_reports_a_trajectory_not_a_running_total():
    from autotube import analytics
    now = common.now_utc()
    from datetime import timedelta
    vids = [{"video_id": f"v{i}", "created_at": (now - timedelta(days=10)).isoformat(),
             "metrics": {"views": 1000, "subs_gained": 4, "subs_lost": 1}} for i in range(5)]
    p = analytics._subs_progress(vids, now)
    assert p["net_90d"] == 15 and p["views_90d"] == 5000
    assert p["per_1k_views"] == pytest.approx(3.0)
    assert p["days_to_500"] and p["days_to_1000"]
    assert p["days_to_1000"] > p["days_to_500"]


def test_subs_progress_does_not_invent_a_projection_without_data():
    from autotube import analytics
    now = common.now_utc()
    p = analytics._subs_progress([], now)
    assert p["days_to_500"] is None and p["days_to_1000"] is None


def test_subs_progress_handles_a_channel_that_is_losing_subscribers():
    from datetime import timedelta

    from autotube import analytics
    now = common.now_utc()
    vids = [{"video_id": "v", "created_at": (now - timedelta(days=5)).isoformat(),
             "metrics": {"views": 1000, "subs_gained": 1, "subs_lost": 9}}]
    p = analytics._subs_progress(vids, now)
    assert p["net_90d"] == -8
    assert p["days_to_500"] is None, "must not project arrival while going backwards"


# ── Previously untested: the CLI surface ────────────────────────────────────
# "doctor" is deliberately excluded: it probes live API endpoints, which made the suite
# take 3.4s longer and depend on network reachability. A test that fails on a train is
# worse than no test. It is exercised manually and by the audit.
@pytest.mark.parametrize("cmd", ["dossier", "report"])
def test_cli_commands_run_offline(cmd, capsys):
    from autotube.__main__ import main
    assert main([cmd]) == 0


def test_dossier_cli_emits_valid_json(capsys):
    from autotube.__main__ import main
    main(["dossier"])
    assert "episodes" in json.loads(capsys.readouterr().out)


def test_originality_audit_is_in_the_analytics_summary():
    import inspect

    from autotube import analytics
    src = inspect.getsource(analytics.run)
    assert '"originality": originality.channel_audit(hist, cfg)' in src
    assert '"subscribers": _subs_progress(' in src
