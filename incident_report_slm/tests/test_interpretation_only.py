"""Regression tests for the one-generated-section contract."""
from copy import deepcopy
import json
import pytest
from wualt_slm import IncidentReporter, IncidentBundle
from wualt_slm.backend import BackendUnavailable
from wualt_slm.contextual_report import build_contextual_view
from wualt_slm.contextual_backend import prompt_evidence
from wualt_slm.contextual_quality import contextual_issues
from test_contextual_reports import fixture, MODEL_FIXTURE
from model_fixtures import draft

class Model:
    def __init__(self, text=None): self.text=text
    def generate_contextual_incident(self, evidence):
        part=deepcopy(MODEL_FIXTURE['interpretation'])
        if self.text: part['text']=self.text
        return json.dumps(draft(part))

def test_model_changes_only_interpretation_not_supplied_template_fields():
    raw=fixture(); raw.update(incident_description='Exact supplied description <tag>.', clarification_required='Exact supplied question?')
    original=deepcopy(raw)
    a=IncidentReporter(backend=Model(),mode='raw_review').generate(raw)
    b=IncidentReporter(backend=Model('A different model paragraph.'),mode='raw_review').generate(raw)
    assert raw==original
    assert a['input']==b['input']==IncidentBundle.model_validate(raw).model_dump(mode='json')
    assert a['view']==b['view']
    assert a['model_sections']==b['model_sections']==['interpretation']
    for name,key in [('description','incident_description'),('clarification','clarification_required')]:
        assert a['narrative'][name]==b['narrative'][name]
        assert a['narrative'][name]['text']==raw[key]
        assert a['narrative'][name]['source']=='supplied'
    assert a['narrative']['interpretation']!=b['narrative']['interpretation']
    assert not a['fallback_enabled'] and 'chat_interpretation' not in a['narrative']

def test_missing_static_description_is_not_filled_by_model():
    result=IncidentReporter(backend=Model(),mode='raw_review').generate(fixture())
    assert result['narrative']['description']['text']=='Not supplied.'
    assert result['narrative']['clarification']['text']=='Not supplied.'
    assert result['input']['button']['state']=='none_recorded'

def test_model_cannot_write_additional_template_sections():
    class Extra:
        def generate_contextual_incident(self,evidence):
            result=draft(MODEL_FIXTURE['interpretation']);result['description']={'text':'Invented report details.','evidence_ids':['M']}
            return json.dumps(result)
    with pytest.raises(BackendUnavailable,match='invalid_contextual_narrative_schema'):
        IncidentReporter(backend=Extra()).generate(fixture())

def test_keyword_negation_and_current_words_survive_prompt_preparation():
    raw=fixture();raw['audio']['transcript'][0]['text']='Nobody is following me. I am running for the bus.'
    raw['audio']['keywords']=['following','running']
    facts=prompt_evidence(build_contextual_view(IncidentBundle.model_validate(raw))['evidence'],'analysis')
    assert next(f['text'] for f in facts if f['id']=='T01')==raw['audio']['transcript'][0]['text']
    keyword=next(f for f in facts if f['id']=='K')
    assert keyword['independent_evidence'] is False
    assert any(f.get('role')=='user' for f in facts)
    assert any(f['id']=='P' for f in facts) and any(f['id']=='M' for f in facts)

def test_urgent_situation_label_alone_does_not_preserve_sos_handling():
    raw=fixture();raw['button']={'state':'sos','at':raw['incident_start']}
    facts=build_contextual_view(IncidentBundle.model_validate(raw))['evidence']
    issues=contextual_issues({'text':'A recorded SOS indicates an urgent situation but the cause is unknown.','evidence_ids':['B']},facts,'interpretation')
    assert 'Preserve urgent handling for the recorded device SOS.' in issues

def test_missing_speech_is_not_given_a_transcription_reliability_claim():
    raw=fixture();raw['audio']={'transcript':[],'transcript_reliability':'unavailable'}
    facts=build_contextual_view(IncidentBundle.model_validate(raw))['evidence']
    issues=contextual_issues({'text':'The transcription reliability is unavailable and the context cannot establish the cause.','evidence_ids':['A']},facts,'interpretation')
    assert any('nonexistent wording' in issue for issue in issues)


def test_transcript_and_button_chronology_are_separate_from_incident_start():
    from wualt_slm.physiology_quality import sentence_issues
    raw=fixture();raw['button']={'state':'sos','at':'2026-10-23T10:00:22+05:30'}
    raw['audio']['transcript'][0]['at']='2026-10-23T10:00:14+05:30'
    facts=build_contextual_view(IncidentBundle.model_validate(raw))['evidence']
    transcript=next(e for e in facts if e['id']=='T01')
    assert transcript['relative_to_button']=='8 seconds before the recorded SOS'
    prepared=prompt_evidence(facts,'analysis')
    assert next(e for e in prepared if e['id']=='T01')['relative_to_button']==transcript['relative_to_button']
    bad={'text':'The speaker asked for the bus after the SOS activation.','evidence_ids':['T01','B']}
    assert any('do not reverse' in s for s in sentence_issues(bad,facts))
    bad['text']='The transcript mentions the bus before the SOS activation.'
    assert not any('do not reverse' in s for s in sentence_issues(bad,facts))


def test_hr_comparison_does_not_falsely_become_an_hrv_comparison():
    from wualt_slm.physiology_quality import sentence_issues
    facts=[{'id':'P','kind':'measurement','text':'HR peak above reference.'}, {'id':'V','kind':'measurement','text':'HRV signal poor; comparison unavailable.','comparison':'unavailable'}]
    part={'text':'The heart rate peak is above the reference, but the HRV reading is unreliable due to poor signal quality.','evidence_ids':['P','V']}
    assert not any('Do not compare HRV' in issue for issue in sentence_issues(part,facts))
    part['text']='HRV is below the reference.'
    assert any('Do not compare HRV' in issue for issue in sentence_issues(part,facts))
