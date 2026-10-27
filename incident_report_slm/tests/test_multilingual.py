"""Multilingual inputs, language-aware checks and the multilingual evaluation runner."""
import copy
import json
import runpy
from pathlib import Path

import httpx
import pytest

import wualt_slm.signal_interpretation as interpretation
from wualt_slm import SignalInterpreter
from wualt_slm.backend import LlamaBackend
from wualt_slm.interpretation_grounding import english_like
from wualt_slm.signal_interpretation import draft_issues, prepare_signals, prepare_audio_signals
from wualt_slm.signal_schema import InterpretationInput, SignalDraft

ROOT = Path(__file__).parents[1]
CASES = {row["scenario"]: row for row in runpy.run_path(str(ROOT / "scripts/multilingual_scenarios.py"))["multilingual_cases"]()}
EXPANDED = {row["scenario"]: row["input"] for row in runpy.run_path(str(ROOT / "scripts/evaluate_signal_interpretations.py"))["cases"]()}


def check(payload, text, ids, section="distress"):
    facts = prepare_signals(InterpretationInput.model_validate(payload))
    if section == "audio":
        facts = prepare_audio_signals(facts)
    draft = SignalDraft.model_validate({"sentences": [{"text": text, "evidence_ids": ids}]})
    return draft_issues(draft, facts, section=section)


def test_multilingual_inputs_are_valid_and_not_english():
    assert len(CASES) == 18
    assert len({row["language"] for row in CASES.values()}) >= 14
    for row in CASES.values():
        InterpretationInput.model_validate(row["input"])
        observed = row["input"]["transcript"] or row["input"]["chat_context"][0]["text"]
        assert not english_like(observed), row["scenario"]


@pytest.mark.parametrize("text,expected", [
    ("My wrist hurts a lot after I bumped it. I need help.", True),
    ("I was scared during the power cut last week. Today I am okay.", True),
    ("Nobody is following me.", True),
    ("Mujhe darr lag raha hai. Darwaza nahi khul raha. Madad kijiye.", False),
    ("Enakku romba tension. Tomorrow interview irukku.", False),
    ("Je vais retrouver mes amis pour déjeuner. Tout va bien.", False),
    ("मैं गिर गया और मेरी कलाई टूट गई है।", False),
])
def test_english_detection(text, expected):
    assert english_like(text) is expected


@pytest.mark.parametrize("text", [
    "The person reports wrist pain after bumping it; this does not confirm a wrist injury, and the state remains uncertain.",
    "The person reports wrist pain, which does not indicate a serious injury, and asks for help that may fit the distress state.",
    "The person reports wrist pain; this does not confirm an injury, though the request for help may fit distress.",
])
def test_negated_injury_with_describing_words_is_accepted(text):
    # Live 0.19.1 run: pain-unchanged-heart failed twice on this check.
    assert not any("confirmed injury" in i for i in check(EXPANDED["pain-unchanged-heart"], text, ["TRANSCRIPT", "STATE"]))


@pytest.mark.parametrize("text", [
    "The person has a wrist injury and asks for help, which fits distress.",
    "The pain suggests an injury, which supports distress.",
    "The person cannot move the wrist and has an injury, which fits distress.",
])
def test_injury_upgrades_are_still_rejected(text):
    assert any("confirmed injury" in i for i in check(EXPANDED["pain-unchanged-heart"], text, ["TRANSCRIPT", "STATE"]))


def test_non_english_injury_report_may_be_attributed_but_not_asserted():
    payload = CASES["hindi-injury-report"]["input"]
    assert not check(payload, "The person says they fell down the stairs and reports a wrist injury, asking for help, which may fit the distress state.", ["TRANSCRIPT", "STATE"])
    issues = check(payload, "The fall caused a wrist injury, which supports the distress state.", ["TRANSCRIPT", "STATE"])
    assert any("attribute it to the person" in i for i in issues)


def test_non_english_pain_is_not_asserted_as_injury():
    issues = check(CASES["telugu-pain-only"]["input"], "The knee injury explains the distress state.", ["TRANSCRIPT", "STATE"])
    assert any("confirmed injury" in i for i in issues)


def test_non_english_speech_adds_a_language_instruction(monkeypatch):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        text = "The person says they were scared last week and are fine today, which does not clearly support the distress state."
        ids = ["TRANSCRIPT", "STATE"] if len(requests) > 1 else ["TRANSCRIPT", "TONE"]
        content = json.dumps({"sentences": [{"text": text if len(requests) > 1 else "The person says they were scared last week and are fine today.", "evidence_ids": ids}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    SignalInterpreter(backend=LlamaBackend()).interpret(CASES["hindi-past-fear"]["input"])
    system = [r["messages"][0]["content"] for r in requests]
    assert all("speech is not in English" in s for s in system)
    assert all(interpretation.LANGUAGE_RULES in s for s in system)

    requests.clear()
    SignalInterpreter(backend=LlamaBackend()).interpret(EXPANDED["past-incident"])
    assert not any("not in English" in r["messages"][0]["content"] for r in requests)


def test_runner_records_rejected_drafts_without_changing_results(monkeypatch):
    runner = runpy.run_path(str(ROOT / "scripts/evaluate_multilingual.py"))
    store = []
    monkeypatch.setattr(interpretation, "draft_issues", interpretation.draft_issues)
    original = runner["capture_rejections"](store)
    facts = prepare_signals(InterpretationInput.model_validate(CASES["telugu-pain-only"]["input"]))
    bad = SignalDraft.model_validate({"sentences": [{"text": "The knee injury explains the distress state.", "evidence_ids": ["TRANSCRIPT", "STATE"]}]})
    good = SignalDraft.model_validate({"sentences": [{"text": "The person says their knee hurts a lot and they cannot walk, which may fit the distress state.", "evidence_ids": ["TRANSCRIPT", "STATE"]}]})
    assert interpretation.draft_issues(bad, facts) == original(bad, facts)
    assert interpretation.draft_issues(good, facts) == []
    assert len(store) == 1 and store[0]["draft"] == "The knee injury explains the distress state." and store[0]["issues"]


# 0.20.1: wording problems found by reading the 0.20.0 live run.

def test_inverted_normal_conclusion_is_rejected():
    payload = CASES["tamil-game-character"]["input"]
    live = "The observations do not clearly support the normal state, as the character's distress is not the speaker's own."
    assert any("reasons describe a calm" in i for i in check(payload, live, ["TRANSCRIPT", "TONE", "STATE"]))
    fixed = "The relaxed tone and the person's statement that they are only playing are consistent with the normal state."
    assert not check(payload, fixed, ["TRANSCRIPT", "TONE", "STATE"])
    # A normal label that the observations genuinely contradict may still be called unsupported.
    legit = "The person says they do not feel safe and asks for help leaving quietly, which does not clearly support the normal state."
    assert not any("reasons describe a calm" in i for i in check(EXPANDED["whispered-help"], legit, ["TRANSCRIPT", "STATE"]))


@pytest.mark.parametrize("text", [
    "The tone suggests the person is tired with high confidence.",
    "A dog barking is detected with high confidence, but this does not confirm an attack.",
    "Their tone reflects worry with a high level of confidence.",
])
def test_high_confidence_wording_is_rejected(text):
    issues = check(CASES["marathi-negation"]["input"], text, ["TRANSCRIPT", "TONE"], "audio")
    assert any("confidence as high" in i for i in issues)


@pytest.mark.parametrize("text", [
    "The person says they are not scared, just tired, and their tone sounds tired.",
    "The weak voice is not confirmed with high certainty.",
    "A possible scream was detected with low confidence.",
])
def test_plain_or_low_certainty_wording_is_accepted(text):
    issues = check(CASES["marathi-negation"]["input"], text, ["TRANSCRIPT", "TONE"], "audio")
    assert not any("confidence as high" in i for i in issues)


@pytest.mark.parametrize("case,text", [
    ("marathi-negation", "The person says they are not scared, and the word 'भीती' (fear) is a keyword."),
    ("hindi-help", "The words 'madad' (help) and 'darr' (fear) are identified as keywords."),
    ("spanish-sister-fear", "The person says they feel 'tranquilo' while their sister is afraid."),
])
def test_original_language_words_in_prose_are_rejected(case, text):
    payload = CASES[case]["input"] if case in CASES else EXPANDED[case]
    assert any("Write only in English" in i for i in check(payload, text, ["TRANSCRIPT", "STATE"]))


def test_english_quotes_and_translations_are_accepted():
    assert not any("Write only in English" in i for i in check(EXPANDED["past-incident"], "The person says they were 'scared' last week and are okay now, which does not support current distress.", ["TRANSCRIPT", "STATE"]))
    assert not any("Write only in English" in i for i in check(CASES["hindi-limited-transcript"]["input"], "The person says 'help... door... closed', but the limited speech quality leaves the meaning uncertain for the distress state.", ["TRANSCRIPT", "STATE"]))


def test_main_evaluation_saves_rejected_drafts(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        # Every first draft says "nearby" (rejected); the rewrite is plain.
        text = "A dog barking was detected nearby." if len(calls) % 2 else "The observations are not enough to explain the supplied state."
        ids = [f["id"] for f in json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]][:1] + ["STATE"]
        content = json.dumps({"sentences": [{"text": text, "evidence_ids": list(dict.fromkeys(ids))}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(interpretation, "draft_issues", interpretation.draft_issues)
    module = runpy.run_path(str(ROOT / "scripts/evaluate_signal_interpretations.py"))
    monkeypatch.setattr("sys.argv", ["evaluate", "--suite", "original", "--model-url", "http://model.test", "--output", str(tmp_path)])
    module["main"]()
    rows = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    drafts = [d for row in rows for d in row["rejected_drafts"]]
    assert drafts and summary["rejected_drafts"] == len(drafts)
    assert any("nearby" in d["draft"] and any("proximity" in i for i in d["issues"]) for d in drafts)
    assert "Rejected drafts" in (tmp_path / "Interpretation_Review.html").read_text(encoding="utf-8")


def test_report_shows_rejected_drafts(tmp_path):
    row = copy.deepcopy(CASES["telugu-pain-only"])
    row["output"] = {"audio_interpretation": None, "distress_interpretation": None,
                     "sections": {n: {"status": "failed", "reason": "grounding_failed", "attempts": 2,
                                      "diagnostics": [{"attempt": 1, "issues": ["x"]}, {"attempt": 2, "issues": ["x"]}]} for n in ("audio", "distress")}}
    row["rejected_drafts"] = [{"section": "distress", "draft": "The knee <injury> explains it.", "issues": ["No injury was reported."]}]
    target = tmp_path / "r.html"
    runpy.run_path(str(ROOT / "scripts/interpretation_review.py"))["save_html"]([row], target)
    html = target.read_text(encoding="utf-8")
    assert "Rejected drafts (1)" in html and "The knee &lt;injury&gt; explains it." in html
