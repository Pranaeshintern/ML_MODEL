"""Scenario integrity and report correctness, without simulated model claims."""
import copy
import json
import runpy
from html.parser import HTMLParser
from pathlib import Path
from wualt_slm.signal_schema import InterpretationInput

MODULE=runpy.run_path(str(Path(__file__).parents[1]/"scripts/evaluate_signal_interpretations.py"))


def test_expanded_inputs_are_valid_and_original_payloads_unchanged():
    expanded=MODULE["cases"]();original=MODULE["regression_cases"]()
    assert len(expanded)==117
    assert len({row["scenario"] for row in expanded})==117
    assert [row["input"] for row in expanded[-37:]]==[row["input"] for row in original]
    for row in expanded:
        InterpretationInput.model_validate(row["input"])


def test_new_scenarios_do_not_inherit_dog_or_sensor_defaults():
    rows=MODULE["cases"]("diverse")
    assert len({row["category"] for row in rows})==10
    assert not any("dog" in json.dumps(row["input"]).lower() for row in rows)
    examples={row["scenario"]:row["input"] for row in rows}
    assert examples["work-deadline"]["events"]==[]
    assert examples["work-deadline"]["hr"] is None
    assert examples["chat-locked-room"]["transcript"] is None
    assert examples["assistant-only-danger"]["chat_context"][0]["role"]=="assistant"
    assert examples["hindi-negation"]["keywords"]["distress"]==["darr"]


def result(text, status="generated", reason=None):
    return {"audio_interpretation":text if status=="generated" else None,
            "distress_interpretation":text if status=="generated" else None,
            "sections":{name:{"status":status,"reason":reason,"attempts":1 if status!="not_available" else 0,"diagnostics":[]} for name in ("audio","distress")}}


def test_report_preserves_input_and_prose_and_escapes_markup(tmp_path):
    row=copy.deepcopy(MODULE["cases"]("diverse")[0])
    row["input"]["transcript"]="<script>alert('input')</script> & தமிழ்"
    row["output"]=result("Exact model prose <b>is data</b> & remains unchanged.")
    target=tmp_path/"report.html";MODULE["save_html"]([row],target)
    page=target.read_text(encoding="utf-8")
    class Capture(HTMLParser):
        def __init__(self):super().__init__();self.active=False;self.data=[]
        def handle_starttag(self,tag,attrs):
            if tag=="pre":self.active=True
        def handle_endtag(self,tag):
            if tag=="pre":self.active=False
        def handle_data(self,value):
            if self.active:self.data.append(value)
    parser=Capture();parser.feed(page)
    assert json.loads("".join(parser.data))==row["input"]
    assert "<script>alert('input')</script>" not in page
    assert "Exact model prose &lt;b&gt;is data&lt;/b&gt; &amp; remains unchanged." in page
    assert row["review_focus"] not in page


def test_report_distinguishes_generation_failure_missing_data_and_pending(tmp_path):
    rows=MODULE["cases"]("diverse")[:3]
    rows[0]["output"]=result(None,"not_available","insufficient_context")
    rows[1]["output"]=result(None,"failed","grounding_failed")
    target=tmp_path/"report.html";MODULE["save_html"](rows,target)
    page=target.read_text(encoding="utf-8")
    assert "not enough reliable information" in page
    assert "generation failure, not evidence that data is missing" in page
    assert "Not run yet. No model interpretation" in page
    assert "2</strong><span>rejected sections" in page
    assert "1</strong><span>not run yet" in page


def test_readability_guard_rejects_processing_metadata():
    from wualt_slm.signal_interpretation import draft_issues,prepare_signals
    from wualt_slm.signal_schema import SignalDraft
    facts=prepare_signals(InterpretationInput.model_validate(MODULE["cases"]("diverse")[0]["input"]))
    for text in ["The backend reports distress.","Distress is not verified ground truth.","Fear with physiological arousal supports distress.","The distress label has 82% confidence."]:
        draft=SignalDraft.model_validate({"sentences":[{"text":text,"evidence_ids":["STATE","TRANSCRIPT"]}]})
        assert any("everyday wording" in issue for issue in draft_issues(draft,facts))


def test_reference_match_does_not_establish_a_clinical_normal_range():
    from wualt_slm.signal_interpretation import draft_issues,prepare_signals
    from wualt_slm.signal_schema import SignalDraft
    row=next(x for x in MODULE["cases"]("diverse") if x["scenario"]=="whispered-help")
    facts=prepare_signals(InterpretationInput.model_validate(row["input"]))
    for text in ["Heart rate is within normal limits, confirming the normal state.","The heart rate and measurements were in the normal range."]:
        draft=SignalDraft.model_validate({"sentences":[{"text":text,"evidence_ids":["STATE","TRANSCRIPT","HR"]}]})
        assert any("clinical normal range" in issue for issue in draft_issues(draft,facts))
    draft=SignalDraft.model_validate({"sentences":[{"text":"The help request is not explained by the normal label. Heart rate matches the supplied reference.","evidence_ids":["STATE","TRANSCRIPT","HR"]}]})
    assert not draft_issues(draft,facts)


def test_reported_pain_is_not_upgraded_to_confirmed_injury():
    from wualt_slm.signal_interpretation import draft_issues,prepare_signals
    from wualt_slm.signal_schema import SignalDraft
    row=next(x for x in MODULE["cases"]("diverse") if x["scenario"]=="fall-self")
    facts=prepare_signals(InterpretationInput.model_validate(row["input"]))
    draft=SignalDraft.model_validate({"sentences":[{"text":"Pain after the fall is indicating a physical injury and distress.","evidence_ids":["STATE","TRANSCRIPT"]}]})
    assert any("confirmed injury" in issue for issue in draft_issues(draft,facts))
