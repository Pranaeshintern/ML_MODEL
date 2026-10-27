"""Regression cases for five human-reviewed failures and valid alternatives."""
import copy
import runpy
from pathlib import Path
import pytest
from wualt_slm.signal_interpretation import draft_issues, prepare_signals, prepare_audio_signals
from wualt_slm.signal_schema import InterpretationInput, SignalDraft

CASES = {row["scenario"]: row["input"] for row in runpy.run_path(str(Path(__file__).parents[1]/"scripts/evaluate_signal_interpretations.py"))["cases"]()}


def check(case, text, ids, section="distress", **changes):
    payload = copy.deepcopy(CASES[case]); payload.update(changes)
    facts = prepare_signals(InterpretationInput.model_validate(payload))
    if section == "audio": facts = prepare_audio_signals(facts)
    draft = SignalDraft.model_validate({"sentences": [{"text": text, "evidence_ids": ids}]})
    return draft_issues(draft, facts, section=section)


@pytest.mark.parametrize("verified", [False, True])
def test_pursuit_is_an_account_even_when_identity_verified(verified):
    assert any("external event" in i for i in check("following-concern", "A person is being followed and is feeling scared.", ["TRANSCRIPT"], "audio", speaker_verified=verified))
    assert not check("following-concern", "The person reports being followed and feeling scared, and asks someone to stay on the phone.", ["TRANSCRIPT"], "audio", speaker_verified=verified)


@pytest.mark.parametrize("text", [
    "The reported injury and concerned tone may support distress.",
    "Their father is injured, which explains the distress.",
    "The pain suggests a fracture and distress.",
])
def test_pain_does_not_license_injury(text):
    assert any("confirmed injury" in i for i in check("fall-other", text, ["STATE", "TRANSCRIPT"]))


@pytest.mark.parametrize("text", [
    "The person reports their father fell and has knee pain, which may explain their concern and the distress label.",
    "The person reports knee pain after a fall; this does not confirm an injury or establish distress.",
    "The person reports a fall, but an injury is not confirmed and distress remains uncertain.",
    "No injury is confirmed; the person reports pain, which may explain distress.",
])
def test_preserve_symptoms_and_noninjury_caveats(text):
    assert not check("fall-other", text, ["STATE", "TRANSCRIPT"])


def test_explicit_injury_account_is_permitted_but_denial_is_not():
    assert not check("fall-other", "The person reports an injury, which may explain distress.", ["STATE", "TRANSCRIPT"], transcript="I injured my knee.")
    assert any("confirmed injury" in i for i in check("fall-other", "The reported injury explains distress.", ["STATE", "TRANSCRIPT"], transcript="I fell but I am not injured."))


def test_crying_for_joy_does_not_become_mixed_emotions():
    assert any("explicitly happy" in i for i in check("tears-of-joy", "The observations suggest a mix of emotions and possible distress.", ["STATE", "TRANSCRIPT", "TONE"]))
    assert not check("tears-of-joy", "The person says they are crying with happiness at seeing someone again; this does not support distress.", ["STATE", "TRANSCRIPT"])
    assert not check("tears-of-joy", "The person reports happiness; the tears do not establish mixed emotions or distress.", ["STATE", "TRANSCRIPT"])
    assert not check("tears-of-joy", "The person reports mixed emotions, leaving distress uncertain.", ["STATE", "TRANSCRIPT"], transcript="I am happy but also sad that you must leave.")


@pytest.mark.parametrize("section,ids", [("audio", ["EVENT1"]), ("distress", ["STATE", "EVENT1"])])
@pytest.mark.parametrize("text", [
    "A scream was detected, but no speech was available and the speaker is unverified.",
    "A scream was detected, but its source and context are unknown.",
    "A scream was detected, but the speaker identity is uncertain.",
    "A scream was detected, but its source is unclear.",
])
def test_uncertain_source_does_not_qualify_low_detection_certainty(section, ids, text):
    assert any("detection itself" in i for i in check("uncertain-sound-label", text, ids, section))


@pytest.mark.parametrize("text", [
    "A possible scream was detected; there is not enough reliable information to explain distress.",
    "The scream detection is uncertain, so it cannot establish distress.",
    "A scream was detected with low confidence, leaving distress uncertain.",
    "A scream-like sound was tentatively detected, without enough context to explain distress.",
])
def test_low_certainty_detection_can_be_explained_naturally(text):
    assert not check("uncertain-sound-label", text, ["STATE", "EVENT1"])


def test_high_certainty_detection_does_not_need_low_certainty_caveat():
    events=copy.deepcopy(CASES["uncertain-sound-label"]["events"]); events[0]["confidence"]=0.95
    assert not check("uncertain-sound-label", "A scream was detected without speech to explain it.", ["EVENT1"], "audio", events=events)


@pytest.mark.parametrize("text", [
    "The person says everything is fine but also mentions feeling unsafe; distress may be present.",
    "The person feels unsafe but cannot speak freely, which supports distress.",
])
def test_chat_conflict_cannot_merge_written_and_spoken_accounts(text):
    assert any("channels" in i for i in check("chat-speech-conflict", text, ["STATE", "TRANSCRIPT", "CHAT1"]))


def test_distinguish_both_channels_and_keep_chat_out_of_audio():
    assert not check("chat-speech-conflict", "The person says everything is fine in the audio, but writes in the chat that they feel unsafe and cannot speak freely; that written account may support distress despite the calm speech.", ["STATE", "TRANSCRIPT", "CHAT1"])
    assert not check("chat-speech-conflict", "The person says everything is fine in a calm tone.", ["TRANSCRIPT", "TONE"], "audio")
    assert any("current user chat" in i for i in check("chat-speech-conflict", "The person says everything is fine, leaving distress unsupported.", ["STATE", "TRANSCRIPT"]))


def test_old_or_assistant_chat_need_not_be_treated_as_current_testimony():
    for scenario in ("old-chat", "assistant-question"):
        assert not any("current user chat" in i or "channels" in i for i in check(scenario, "The person reports their situation; distress remains uncertain.", ["STATE", "TRANSCRIPT"]))
