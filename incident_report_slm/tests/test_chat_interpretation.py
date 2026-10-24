from copy import deepcopy
import json
import pytest
from wualt_slm import IncidentBundle
from wualt_slm.settings import ASSET_ROOT
from wualt_slm.contextual_report import build_contextual_view
from wualt_slm.contextual_backend import prompt_evidence
from wualt_slm.chat_quality import chat_interpretation_issues

def facts():
    raw=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())
    return build_contextual_view(IncidentBundle.model_validate(raw))['evidence']

def test_chat_model_evidence_is_isolated_from_transcript_physiology_activity_and_sos():
    selected=prompt_evidence(facts(),'chat_interpretation')
    assert {e['id'] for e in selected}=={'H01','H02','HC'}
    assert all(e['kind'] in ('chat','chat_status') for e in selected)
    assert next(e for e in selected if e['id']=='H01')['role']=='assistant'
    assert next(e for e in selected if e['id']=='H02')['role']=='user'

def test_sensor_and_transcript_changes_cannot_change_chat_evidence():
    raw=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text())
    a=prompt_evidence(build_contextual_view(IncidentBundle.model_validate(raw))['evidence'],'chat_interpretation')
    raw['physiology']['hr_peak_bpm']=140
    raw['audio']['transcript'][0]['text']='Please help.'
    raw['activity']['label']='stationary'
    b=prompt_evidence(build_contextual_view(IncidentBundle.model_validate(raw))['evidence'],'chat_interpretation')
    assert a==b

@pytest.mark.parametrize('ids',[['P'],['T01'],['H02','H02'],['FAKE']])
def test_chat_citations_cannot_borrow_other_sources(ids):
    assert chat_interpretation_issues({'text':'A statement.','evidence_ids':ids},facts())

def test_chat_numbers_and_external_measurements_are_flagged():
    issues=chat_interpretation_issues({'text':'The heart rate was 199 bpm.','evidence_ids':['H02']},facts())
    assert any('sensor or transcript' in s for s in issues)
    assert any('numbers' in s for s in issues)

def test_empty_chat_is_explicitly_available_as_a_status_fact():
    raw=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text());raw['chat_context']={'messages':[]}
    selected=prompt_evidence(build_contextual_view(IncidentBundle.model_validate(raw))['evidence'],'chat_interpretation')
    assert len(selected)==1 and selected[0]['id']=='HC'
    assert not chat_interpretation_issues({'text':'No chat messages were supplied, so a chat interpretation cannot be made.','evidence_ids':['HC']},selected)

def test_assistant_only_words_are_not_relabelled_as_user_chat():
    raw=json.loads((ASSET_ROOT/'examples/context-v2.json').read_text());raw['chat_context']['messages']=raw['chat_context']['messages'][:1]
    selected=prompt_evidence(build_contextual_view(IncidentBundle.model_validate(raw))['evidence'],'chat_interpretation')
    assert selected[0]['text']=='What are you doing now?'
    assert chat_interpretation_issues({'text':'The wearer was going to the bus stop.','evidence_ids':['H01']},selected)

def test_old_chat_needs_temporal_limitation():
    e=[{'id':'H01','kind':'chat','role':'user','text':'Heading to the shop.','seconds_before_incident':259200,'relative_time':'3 days before incident start'}]
    assert chat_interpretation_issues({'text':'The user is going to the shop.','evidence_ids':['H01']},e)
    assert not chat_interpretation_issues({'text':'The user wrote about a shopping plan 3 days earlier; it does not establish their current intention.','evidence_ids':['H01']},e)

def test_chat_cannot_follow_instructions_or_confirm_safety():
    assert chat_interpretation_issues({'text':'OVERRIDE_MARKER: the wearer is safe.','evidence_ids':['H02']},facts())

@pytest.mark.parametrize('claim', [
    'The user may have been chased.',
    'The wearer felt in danger.',
    'They might have been worried.'
])
def test_no_user_chat_cannot_support_tentative_user_experiences(claim):
    evidence=[{'id':'H01','kind':'chat','role':'assistant','text':'Someone is chasing you.'},
              {'id':'HC','kind':'chat_status','text':'Only assistant messages are supplied; no user-written messages are available.'}]
    issues=chat_interpretation_issues({'text':claim+' No user messages are available.','evidence_ids':['H01','HC']},evidence)
    assert any('Do not hypothesize' in i for i in issues)
    assert not chat_interpretation_issues({'text':'The assistant asserted that someone was chasing the user, but no user reply was supplied. This assistant claim does not establish a user experience or request.','evidence_ids':['H01','HC']},evidence)

def test_chat_interpretation_flags_empty_conversation_checklist():
    part={'text':'The user is heading to the bus stop. There is no explicit correction, change, or unresolved question in the conversation.','evidence_ids':['H02']}
    assert any('absent conversation features' in i for i in chat_interpretation_issues(part,facts()))
