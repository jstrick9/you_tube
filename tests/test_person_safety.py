"""Living people: public figures on professional angles only (autotube/safety.check_person)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import safety  # noqa: E402
from autotube.common import load_config  # noqa: E402

CFG = load_config()


class LLM:
    """A correct reviewer. Also records whether it was called at all."""
    def __init__(self, **kw):
        self.kw, self.calls = kw, 0

    def json(self, system, user, **k):
        self.calls += 1
        return {"is_public_figure": self.kw.get("pub", True), "is_adult": self.kw.get("adult", True),
                "angle_is_professional": self.kw.get("prof", True),
                "allowed": self.kw.get("allowed", True), "reason": "r",
                "safe_angle": "their new album"}


class Down:
    def json(self, *a, **k):
        raise RuntimeError("all providers down")


# ── the public-figure line ────────────────────────────────────────────────────
def test_public_figure_on_professional_work_is_allowed():
    ok, _, angle = safety.check_person("Taylor Swift announces an album", "singer", CFG, LLM())
    assert ok and angle


def test_private_individual_is_refused_even_when_the_model_says_yes():
    """The trend scanner will surface 'the guy who went viral'. He has no public role to discuss
    and is protected by YouTube's privacy and harassment policies."""
    ok, reason, _ = safety.check_person("Man who went viral at a concert", "a bystander",
                                        CFG, LLM(pub=False, allowed=True))
    assert not ok and "private individual" in reason


def test_minors_are_refused_even_when_the_model_says_yes():
    ok, reason, _ = safety.check_person("Teen prodigy signs a deal", "16-year-old",
                                        CFG, LLM(adult=False, allowed=True))
    assert not ok and "minor" in reason


def test_non_professional_angle_is_refused_even_when_the_model_says_yes():
    ok, reason, _ = safety.check_person("A performer at home", "lifestyle",
                                        CFG, LLM(prof=False, allowed=True))
    assert not ok and "professional" in reason


# ── the cheap deterministic pass ──────────────────────────────────────────────
def test_blocked_angles_never_reach_the_model():
    for subject in ["Actor hospitalised after collapse", "Rapper sued over a sample",
                    "Star's shock divorce", "CEO net worth revealed", "Singer's cancer diagnosis",
                    "Athlete arrested overnight", "Actor dead at 54", "Star slammed by fans",
                    "Rumoured new romance"]:
        llm = LLM()
        ok, reason, _ = safety.check_person(subject, "", CFG, llm)
        assert not ok, subject
        assert llm.calls == 0, f"{subject!r} should be rejected before paying for a model call"
        assert "blocked angle" in reason


def test_ordinary_professional_news_does_reach_the_model():
    llm = LLM()
    safety.check_person("Director announces a new film", "cinema", CFG, llm)
    assert llm.calls == 1


# ── failing closed ────────────────────────────────────────────────────────────
def test_no_classifier_means_no_video():
    assert not safety.check_person("Taylor Swift", "singer", CFG, None)[0]


def test_classifier_outage_means_no_video():
    ok, reason, _ = safety.check_person("Taylor Swift", "singer", CFG, Down())
    assert not ok and "unavailable" in reason


# ── config ────────────────────────────────────────────────────────────────────
def test_public_figure_requirement_can_be_relaxed_by_config():
    cfg = {"safety": {"person": {"require_public_figure": False}}}
    assert safety.check_person("Someone", "", cfg, LLM(pub=False, allowed=True))[0]


def test_person_config_defaults_survive_yaml_nulls():
    p = safety.person_config({"safety": {"person": {"blocked_angles": None}}})
    assert p["blocked_angles"] == safety.PERSON_BLOCKED_ANGLES
    assert safety.person_config({})["require_public_figure"] is True


# ── script level ──────────────────────────────────────────────────────────────
def test_speculation_is_banned_on_every_topic():
    """A channel whose premise is 'Sounds fake. It's proven.' may never say 'allegedly'."""
    for phrase in ["She reportedly recorded it in secret.", "The eruption allegedly began at dawn.",
                   "Sources say the launch slipped.", "Fans think it is a hint."]:
        assert safety.check_script("A Title", phrase, CFG), phrase


def test_blocked_angles_in_narration_are_caught_only_for_people():
    assert safety.check_script("T", "Her divorce changed everything.", CFG, about_person=True)
    # the same word about a non-person topic is not a person-angle problem
    assert not safety.check_script("T", "Divorce rates fell that decade.", CFG, about_person=False)


def test_a_clean_professional_line_passes():
    assert safety.check_script("Her Record-Breaking Season",
                               "She won eleven titles in a single season.", CFG,
                               about_person=True) == []
