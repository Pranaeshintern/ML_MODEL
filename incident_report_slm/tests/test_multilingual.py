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
