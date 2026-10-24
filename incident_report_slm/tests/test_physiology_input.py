from copy import deepcopy
import json
import pytest
from pydantic import ValidationError
from wualt_slm.incident_schema import IncidentBundle
from wualt_slm.settings import ASSET_ROOT
from wualt_slm.contextual_report import build_contextual_view
from wualt_slm.physiology_quality import sentence_issues


def fixture():
    d=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text(encoding='utf-8'))
    d['physiology']['hrv']={'metric':'rmssd','value_ms':28,'window_start':d['observation_window']['start'],
        'window_end':d['observation_window']['end'],'signal_quality':'usable','artifact_corrected':True,
        'reference':{'metric':'rmssd','value_ms':52,'window_seconds':60,'at':'2026-10-23T09:55:00+05:30','comparable_context':True}}
    return d


@pytest.mark.parametrize('value,expected',[(0,'below'),(28,'below'),(52,'equal'),(70,'above')])
def test_hrv_comparison_preserves_zero_and_direction(value,expected):
    d=fixture();d['physiology']['hrv']['value_ms']=value
    v=next(e for e in build_contextual_view(IncidentBundle.model_validate(d))['evidence'] if e['id']=='V')
    assert v['comparison']==expected and v['value_ms']==value


@pytest.mark.parametrize('change',[
    lambda h:h.update(signal_quality='unreliable'),lambda h:h.update(artifact_corrected=None),
    lambda h:h.update(reference=None),lambda h:h['reference'].update(metric='sdnn'),
    lambda h:h['reference'].update(window_seconds=300),lambda h:h['reference'].update(comparable_context=False),
    lambda h:h.update(window_start='2026-10-23T09:56:00+05:30',window_end='2026-10-23T09:57:00+05:30'),
])
def test_incompatible_hrv_is_recorded_but_not_compared(change):
    d=fixture();change(d['physiology']['hrv'])
    facts=build_contextual_view(IncidentBundle.model_validate(d))['evidence'];v=next(e for e in facts if e['id']=='V')
    assert v['comparison']=='unavailable' and v['comparison_limitations']
    assert any('Do not compare HRV' in s for s in sentence_issues({'text':'HRV is lower than the reference.','evidence_ids':['V']},facts))


@pytest.mark.parametrize('change',[
    lambda d:d['physiology']['hrv'].update(value_ms=-1),lambda d:d['physiology']['hrv'].update(metric='unknown'),
    lambda d:d['physiology']['hrv'].update(window_end=d['observation_window']['start']),
    lambda d:d['physiology']['hrv'].update(window_end='2026-10-23T11:00:00+05:30'),
    lambda d:d['physiology']['hrv']['reference'].update(at='2026-10-23T10:00:30+05:30'),
    lambda d:d['audio'].update(keywords=[' ']),lambda d:d['audio'].update(keywords=['Help','help']),
    lambda d:d['physiology'].update(other_measurements=[{'name':'spo2_percent','value':120,'at':d['incident_start']}]),
])
def test_invalid_backend_payload_is_rejected(change):
    d=fixture();change(d)
    with pytest.raises(ValidationError):IncidentBundle.model_validate(d)


def test_keywords_are_never_independent_corroboration():
    d=fixture();d['audio']['keywords']=['bus','attack']
    v=build_contextual_view(IncidentBundle.model_validate(d));k=next(e for e in v['evidence'] if e['id']=='K')
    assert k['independent_evidence'] is False and k['matched_transcript_ids']==['T01']
    assert v['keywords']==['bus','attack']
    assert d['audio']['transcript'][0]['text']==fixture()['audio']['transcript'][0]['text']


def test_keywords_can_arrive_without_transcript_but_remain_unverified():
    d=fixture();d['audio']={'transcript':[],'transcript_reliability':'unavailable','keywords':['help']}
    k=next(e for e in build_contextual_view(IncidentBundle.model_validate(d))['evidence'] if e['id']=='K')
    assert k['matched_transcript_ids']==[] and 'unverified' in k['limitation']


def test_sentence_cannot_borrow_number_from_uncited_fact():
    facts=[{'id':'P','kind':'measurement','text':'128 bpm'},{'id':'V','kind':'measurement','text':'28 ms'}]
    assert any('own citations' in s for s in sentence_issues({'text':'HRV is 128 ms.','evidence_ids':['V']},facts))


def test_unidentified_speech_does_not_become_wearer_testimony():
    facts=[{'id':'T01','kind':'transcript','text':'Help me.'}]
    assert any('Attribute unidentified' in s for s in sentence_issues({'text':'The wearer asked for help.','evidence_ids':['T01']},facts))


@pytest.mark.parametrize('comparison,text',[
    ('above','HRV is below the reference.'),('below','HRV is higher than the reference.'),
    ('equal','RMSSD is reduced relative to the reference.')])
def test_hrv_comparison_direction_is_checked(comparison,text):
    facts=[{'id':'V','kind':'measurement','text':'HRV comparison','comparison':comparison}]
    assert any('comparison direction' in s for s in sentence_issues({'text':text,'evidence_ids':['V']},facts))


def test_sentence_provenance_must_match_the_returned_paragraph():
    from wualt_slm.incident_schema import ContextualDraft
    part={'text':'Recorded context.','evidence_ids':['P']}
    raw={'description':part,'interpretation':part,'clarification':part,'chat_interpretation':part,
         'interpretation_sentences':[{'text':'Different text.','evidence_ids':['P'],'kind':'interpretation'}]}
    with pytest.raises(ValidationError,match='must match'):
        ContextualDraft.model_validate(raw)


@pytest.mark.parametrize('text',[
    'The heart rate suggests stress, but the cause cannot be confirmed.',
    'The denial of distress creates a discrepancy with the physiological response.',
    'Dizziness does not align with being stationary.',
])
def test_observed_semantic_regressions_are_flagged_without_rewriting(text):
    facts=[{'id':'P','kind':'measurement','text':'Heart-rate difference.'}]
    part={'text':text,'evidence_ids':['P']}
    assert any('Remove unsupported' in s for s in sentence_issues(part,facts))
    assert part['text']==text


def test_hrv_and_identity_errors_outweigh_presentation_improvements():
    from wualt_slm.contextual_quality import contextual_issue_rank
    factual=['Do not compare HRV with this reference; its measurement context is incompatible or unverified.']
    style=['Shorten this section without repeating facts or qualifications.',
           'Keep evidence IDs outside the report prose.',
           'Acknowledge the supplied transcription reliability in the interpretation.']
    assert contextual_issue_rank(style) < contextual_issue_rank(factual)
