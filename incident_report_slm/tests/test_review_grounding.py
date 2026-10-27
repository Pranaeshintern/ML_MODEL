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


def test_conclusion_can_cite_already_qualified_detection_without_repeating_it():
    facts=prepare_signals(InterpretationInput.model_validate(CASES["hard-braking"]))
    draft=SignalDraft.model_validate({"sentences":[
        {"text":"The person reports a bang after sudden braking; a possible collision was detected.","evidence_ids":["TRANSCRIPT","EVENT1"]},
        {"text":"The observations are insufficient to confirm stress because the event details remain uncertain.","evidence_ids":["STATE","EVENT1"]},
    ]})
    assert not draft_issues(draft,facts)


def test_score_repair_cannot_leave_a_meaningless_confidence_fragment():
    issues=check("dizzy-after-standing", "The person reports dizziness. The provided stress label has a confidence, but the observations are limited.", ["STATE","TRANSCRIPT"])
    assert any("entire clause" in issue for issue in issues)


def test_chat_only_paragraph_can_establish_written_channel_once():
    facts=prepare_signals(InterpretationInput.model_validate(CASES["chat-spanish"]))
    draft=SignalDraft.model_validate({"sentences":[
        {"text":"The person reports being trapped in an elevator and feeling afraid, asking for help.","evidence_ids":["CHAT1"]},
        {"text":"This written message may support distress because it describes fear and asks for assistance.","evidence_ids":["CHAT1","STATE"]},
    ]})
    assert not draft_issues(draft,facts)
    draft.sentences[1].text="This statement may support distress because it describes fear and asks for assistance."
    assert any("channels" in issue for issue in draft_issues(draft,facts))


@pytest.mark.parametrize("text", [
    "The person asks about a noise, but the sound's nature is unclear.",
    "The person asks about a noise, and the sound detection for a gunshot had low confidence.",
])
def test_natural_sound_uncertainty_does_not_require_one_fixed_phrase(text):
    assert not check("uncertain-gunshot", text, ["TRANSCRIPT","EVENT1"], "audio")


# v0.18.0 human review: four interpretations needed correction.

def test_heart_comparisons_are_not_a_trend_across_the_window():
    reviewed = "The person's heart rate and heart rate variability remained consistent during the observation window, and they were classified as walking, which aligns with normal activity."
    assert any("one measurement compared with one reference" in i for i in check("walking-chat-context", reviewed, ["HR", "HRV", "ACTIVITY", "STATE"]))
    faithful = "The person was classified as walking, and their heart rate and variation between heartbeats were unchanged from the earlier reference, which fits the normal state."
    assert not check("walking-chat-context", faithful, ["HR", "HRV", "ACTIVITY", "STATE"])


@pytest.mark.parametrize("section,ids", [("audio", ["TRANSCRIPT", "TONE"]), ("distress", ["TRANSCRIPT", "TONE", "STATE"])])
def test_game_character_call_for_help_stays_with_the_character(section, ids):
    reviewed = "A person is playing a game and shouting for help, but they clarify they are just playing, and the relaxed tone fits the normal state."
    assert any("character is calling for help" in i for i in check("game-help-quote", reviewed, ids, section))
    faithful = "The person says their game character is shouting help and that they are just playing, and the relaxed tone fits the normal state."
    assert not check("game-help-quote", faithful, ids, section)


def test_past_fear_does_not_support_current_distress():
    facts = prepare_signals(InterpretationInput.model_validate(CASES["past-incident"]))
    reviewed = SignalDraft.model_validate({"sentences": [
        {"text": "The person reported being scared during a power cut last week but says they are okay today.", "evidence_ids": ["TRANSCRIPT"]},
        {"text": "The transcript mentions scared, which aligns with the distress label, though the speaker is not verified.", "evidence_ids": ["TRANSCRIPT", "KEYWORDS", "STATE"]},
    ]})
    issues = draft_issues(reviewed, facts)
    assert any("Past fear is not current distress" in i for i in issues)
    assert any("does not clearly support the supplied distress" in i for i in issues)
    faithful = SignalDraft.model_validate({"sentences": [
        {"text": "The person says they were scared during a power cut last week and are okay today, so this account does not support current distress.", "evidence_ids": ["TRANSCRIPT", "STATE"]},
        {"text": "Their heart rate was above the reference and the variation between heartbeats was below it, but these readings alone do not establish distress.", "evidence_ids": ["HR", "HRV", "STATE"]},
    ]})
    assert not draft_issues(faithful, facts)


def test_usable_transcript_gets_no_reliability_caveat_and_calm_message_is_weighed():
    facts = prepare_signals(InterpretationInput.model_validate(CASES["chat-recovery"]))
    reviewed = SignalDraft.model_validate({"sentences": [
        {"text": "In the audio, the person said they were frightened when the lights went out, but the transcript's reliability is not explicitly confirmed.", "evidence_ids": ["TRANSCRIPT"]},
        {"text": "In a written message, the person wrote that the lights are back and they feel calm now, which may indicate a resolution.", "evidence_ids": ["CHAT1"]},
        {"text": "The overall situation is assessed as distress based on the provided label.", "evidence_ids": ["STATE"]},
    ]})
    issues = draft_issues(reviewed, facts)
    assert any("marked usable" in i for i in issues)
    assert any("does not clearly support the supplied distress" in i for i in issues)
    faithful = SignalDraft.model_validate({"sentences": [
        {"text": "In the audio, the person said they were frightened when the lights went out.", "evidence_ids": ["TRANSCRIPT"]},
        {"text": "In a written message, the person wrote that the lights are back and they feel calm now, so the current account does not clearly support the distress state.", "evidence_ids": ["CHAT1", "STATE"]},
    ]})
    assert not draft_issues(faithful, facts)
    limited = check("chat-recovery", "In the audio, the person said they were frightened, though the transcript reliability is limited.", ["TRANSCRIPT"], "audio", transcript_reliability="limited")
    assert not any("marked usable" in i for i in limited)
    for doubt in ("The reliability of the transcription is not confirmed.", "The transcription may be inaccurate.", "The accuracy of the transcript is uncertain."):
        assert any("marked usable" in i for i in check("chat-recovery", "The person said they were frightened. " + doubt, ["TRANSCRIPT"], "audio"))


@pytest.mark.parametrize("text", [
    "The person asks for help to keep a dog away; the reliability of the tone and the barking detection is not established.",
    "The person asks for help to keep a dog away, but the tone and sound labels are unreliable and their reliability is unknown.",
    "The person asks for help to keep a dog away, while the frightened tone label has no reliable confidence.",
])
def test_doubting_zero_confidence_labels_is_not_a_transcript_reliability_caveat(text):
    # Live 0.19.1 run: the broader check rejected both audio drafts for this case.
    assert not any("marked usable" in i for i in check("zero-confidence", text, ["TRANSCRIPT", "TONE", "EVENT1"], "audio"))
