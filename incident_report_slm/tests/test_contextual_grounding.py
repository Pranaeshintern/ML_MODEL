import json
import pytest
from wualt_slm import IncidentBundle
from wualt_slm.settings import ASSET_ROOT
from wualt_slm.contextual_report import build_contextual_view
from wualt_slm.contextual_backend import prompt_evidence, signal_constraints
from wualt_slm.contextual_quality import contextual_issues, positive_sos_claim, dismisses_sos


def fixture():
    return json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())


def test_source_text_and_temporal_metadata_are_not_merged():
    raw=fixture();raw['chat_context']['messages'][-1]['at']='2026-10-20T09:59:00+05:30'
    view=build_contextual_view(IncidentBundle.model_validate(raw))
    transcript=next(e for e in view['evidence'] if e['kind']=='transcript')
    chat=next(e for e in view['evidence'] if e.get('role')=='user')
    assert transcript['text']==raw['audio']['transcript'][0]['text']
    assert chat['text']==raw['chat_context']['messages'][-1]['text']
    assert 'days before' in chat['relative_time']
    assert transcript['at'] in view['transcript']
    assert chat['at'] in view['chat_context']


def test_assistant_only_claims_are_retained_in_record_but_not_presented_as_incident_evidence():
    raw=fixture();raw['chat_context']['messages']=[{'role':'assistant','at':'2026-10-23T09:59:00+05:30','text':'An attacker is following you.'}]
    view=build_contextual_view(IncidentBundle.model_validate(raw))
    assert 'An attacker is following you.' in view['chat_context']
    assert 'An attacker' not in json.dumps(prompt_evidence(view['evidence'],'analysis'))
    raw['chat_context']['messages'].append({'role':'user','at':'2026-10-23T10:00:00+05:30','text':'No, that is wrong.'})
    view=build_contextual_view(IncidentBundle.model_validate(raw))
    assert 'An attacker' in json.dumps(prompt_evidence(view['evidence'],'analysis'))


@pytest.mark.parametrize('text,expected',[
    ('A recorded SOS was detected.',True),('An SOS was triggered.',True),
    ('No recorded SOS was detected.',False),('An SOS was not recorded.',False),
])
def test_spoken_help_cannot_be_relabelled_as_a_device_press(text,expected):
    assert positive_sos_claim(text.lower()) is expected


@pytest.mark.parametrize('text,expected',[
    ('Cancel the SOS.',True),('Do not cancel the SOS.',False),
    ('The ordinary explanation does not cancel an SOS.',False),('The SOS is not required.',True),
])
def test_sos_negation_is_not_mistaken_for_cancellation(text,expected):
    assert dismisses_sos(text.lower()) is expected


def test_no_reference_disallows_elevated_claim_and_preserves_exact_measurement():
    raw=fixture();raw['physiology']={'hr_peak_bpm':128}
    facts=build_contextual_view(IncidentBundle.model_validate(raw))['evidence']
    issues=contextual_issues({'text':'The heart rate peaked at 128 bpm, which is elevated.','evidence_ids':['P']},facts,'description')
    assert any('No heart-rate reference' in issue for issue in issues)
    assert 'unavailable' in signal_constraints(facts)


def test_clarification_cannot_delegate_interpretation_back_to_the_wearer():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    issues=contextual_issues({'text':'What is the exact relationship between heart rate and running?','evidence_ids':['P','M']},facts,'clarification')
    assert any('concrete action' in issue for issue in issues)


def test_old_fix_is_not_described_as_current_even_without_last_known_flag():
    raw=fixture();raw['location']['fixed_at']='2026-10-23T08:00:00+05:30'
    fact=next(e for e in build_contextual_view(IncidentBundle.model_validate(raw))['evidence'] if e['id']=='G')
    assert fact['seconds_before_incident']>7200
    assert 'does not verify the current position' in fact['text']


def test_timeline_orders_absolute_instants_across_timezone_offsets():
    facts=[{'id':'H01','kind':'chat','role':'user','text':'First','at':'2026-10-23T09:00:00+05:30'},
           {'id':'H02','kind':'chat','role':'user','text':'Second','at':'2026-10-23T04:00:00+00:00'}]
    prepared={f['id']:f for f in prompt_evidence(facts,'analysis')}
    assert prepared['H01']['sequence']==1
    assert prepared['H02']['sequence']==2


def test_missing_location_is_not_prompted_as_an_incident_explanation():
    facts=[{'id':'G','kind':'location','text':'No location supplied.','available':False}]
    assert prompt_evidence(facts,'analysis')==[]


def test_relative_age_numbers_are_supported_without_inventing_measurements():
    facts=[{'id':'H01','text':'Earlier plan.','kind':'chat','relative_time':'3.0 days before incident start'}]
    issues=contextual_issues({'text':'The user wrote the plan 3 days earlier.','evidence_ids':['H01']},facts,'description')
    assert not any('Use only numbers' in i for i in issues)


def test_pre_event_reference_does_not_become_a_resting_baseline():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    issues=contextual_issues({'text':'The peak was 128 bpm versus the usual 82 bpm.','evidence_ids':['P']},facts,'description')
    assert any('pre-event reference distinct' in i for i in issues)


def test_supplied_location_is_not_reported_as_missing():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    issues=contextual_issues({'text':'There is no location fix.','evidence_ids':['G']},facts,'description')
    assert any('location fix is supplied' in i for i in issues)


def test_unverified_causal_explanation_is_flagged_without_rewriting():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    part={'text':'The heart rate change directly reflects running, not distress.','evidence_ids':['P','M']}
    original=part['text']
    assert any('heart-rate cause' in i for i in contextual_issues(part,facts,'interpretation'))
    assert part['text']==original


def test_clarification_prompt_excludes_measurement_and_location_fields():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    selected=prompt_evidence(facts,'clarification')
    assert not {'P','G','A'} & {e['id'] for e in selected}
    assert {'T01','H02','M'} <= {e['id'] for e in selected}


def test_interpretation_receives_physiology_with_explicit_quality_and_reference():
    facts=build_contextual_view(IncidentBundle.model_validate(fixture()))['evidence']
    assert 'P' in {e['id'] for e in prompt_evidence(facts,'analysis')}
    assert next(e for e in facts if e['id']=='P')['signal_quality']=='unknown'
    assert 'P' in {e['id'] for e in prompt_evidence(facts,'description')}


def test_sos_question_must_address_assistance_needed_now():
    raw=fixture();raw['button']={'state':'sos','at':'2026-10-23T10:00:20+05:30'}
    facts=build_contextual_view(IncidentBundle.model_validate(raw))['evidence']
    assert prompt_evidence(facts,'clarification')[0]['id']=='B'
    issues=contextual_issues({'text':'Are you still heading to the bus stop?','evidence_ids':['B']},facts,'clarification')
    assert any('assistance needed now' in i for i in issues)
