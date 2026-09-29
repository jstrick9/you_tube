"""Offline unit tests for the safety-critical logic.  Run:  python -m pytest -q  (or python tests/test_core.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube.llm import _extract_json  # noqa: E402
from autotube.media import _license_ok  # noqa: E402
from autotube.research import unsupported_numbers  # noqa: E402
from autotube.strategy import Strategy  # noqa: E402
from autotube.trends import _blocked, is_living_person  # noqa: E402
from autotube.common import load_config  # noqa: E402

CFG = load_config()


def test_fact_check_numbers():
    src = "The population was 21 in 1941 and just over 830 birds as of 2025. It weighs 7.5 kg."
    assert unsupported_numbers("Only 21 left in 1941, now 830 birds!", src) == []
    assert unsupported_numbers("Now there are 9,000 of them", src) == ["9000"]
    assert unsupported_numbers("3 amazing facts", src) == []          # small structural numbers ok


def test_json_repair_truncated():
    raw = '```json\n{"title": "X", "segments": [{"text": "a"}, {"text": "b", "keywords": ["c'
    o = _extract_json(raw)
    assert o["title"] == "X" and o["segments"][0]["text"] == "a"


def test_blocklist():
    bl = CFG["compliance"]["blocked_topics"]
    assert _blocked("Mass shooting in city", bl) == "shooting"
    assert _blocked("Whooping crane migration", bl) is None
    assert _blocked("Warsaw zoo", bl) is None                         # word boundary: 'war' ≠ 'warsaw'


def test_living_person():
    assert is_living_person({"description": "American model and actress (born 1966)"})
    assert not is_living_person({"description": "Species of bird"})
    assert not is_living_person({"description": "English naturalist (1809–1882)"})


def test_licenses():
    allowed = CFG["compliance"]["allowed_licenses"]
    assert _license_ok("CC BY-SA 4.0", allowed)
    assert _license_ok("Public domain", allowed)
    assert _license_ok("CC0", allowed)
    assert not _license_ok("CC BY-NC 2.0", allowed)
    assert not _license_ok("Fair use", allowed)
    assert not _license_ok("", allowed)




def test_bandit_learns():
    import random
    random.seed(1)
    s = Strategy(CFG)
    s.state = {"arms": {}, "updates": 0}
    s.__init__.__func__  # noqa
    s = Strategy.__new__(Strategy)
    s.cfg, s.explore, s.state = CFG, 0.0, {"arms": {}, "updates": 0}
    for dim, opts in s.options().items():
        s.state["arms"][dim] = {o: {"a": 1.0, "b": 1.0, "n": 0} for o in opts}
    for _ in range(40):
        s.update({"format": "sounds_fake"}, 0.9)
        s.update({"format": "scale_shock"}, 0.1)
    picks = [s.sample("format") for _ in range(200)]
    assert picks.count("sounds_fake") > picks.count("scale_shock") * 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("✓", name)
