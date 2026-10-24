from collections import Counter
from datetime import datetime, timezone
import os
import re
from uuid import uuid4
from pydantic import ValidationError
from .backend import BackendUnavailable, RUNTIME_LOCK
from .incident_facts import build_incident_view
from .incident_schema import IncidentDraft

PROMPT_VERSION='wualt-incident-v3'
NUMBERS=re.compile(r'(?<![A-Za-z])\d+(?:\.\d+)?')


def validate_narrative(part, name, data, view):
    issues=[];by_id={e['id']:e['text'] for e in view['evidence']}
    if len(set(part.evidence_ids))!=len(part.evidence_ids) or any(i not in by_id for i in part.evidence_ids):
        return ['unknown_or_repeated_evidence']
    source=' '.join(by_id[i] for i in part.evidence_ids)
    if Counter(NUMBERS.findall(part.text))-Counter(NUMBERS.findall(source)):issues.append('unsupported_number')
    times=re.findall(r'\b\d{2}:\d{2}:\d{2}\b',part.text)
    if any(time not in source for time in times):issues.append('unsupported_timestamp')
    if len(part.text.split())>(85 if name=='description' else 110):issues.append('narrative_too_long')
    if re.search(r'[<>]|\{\{|\}\}',part.text):issues.append('unexpected_markup')
    if re.search(r'\b(diagnosed|panic attack|heart attack|assault confirmed|wearer is safe|no danger|accidental press|false alarm)\b',part.text,re.I):issues.append('unsupported_conclusion')
    if re.search(r'\b(man|woman|male|female|boy|girl)\b|\b(a person|the person|person\x27s)\b',part.text,re.I):issues.append('unsupported_identity_inference')
    if re.search(r'\b(normal|abnormal|healthy)\b',part.text,re.I):issues.append('unsupported_clinical_classification')
    if not view['gps']['last_known'] and re.search(r'last.known|stale location',part.text,re.I):issues.append('unsupported_location_status')
    if re.search(r'\b(SOS handling|emergency response|alert|dispatch|notification|responder) (was |is |has been |had been )?(not )?(recorded|sent|initiated|cancelled|dispatched|completed)',part.text,re.I):issues.append('unsupported_response_claim')
    if data.button.state!='sos' and re.search(r'\bSOS\b',part.text,re.I):issues.append('unsupported_sos_action')
    audio_missing=r'missing audio|no audio|audio (is |was )?(not received|unavailable)|audio and speaker data (are |were )?unavailable'
    if data.audio.received and re.search(audio_missing,part.text,re.I):issues.append('unsupported_missing_audio')
    if data.location and re.search(r'no (fresh )?location\b[^.!?]{0,50}\b(supplied|provided)|location (is |was )?(missing|unavailable)',part.text,re.I):
        # A statement about a missing fresh fix is valid only for last-known data.
        if not (view['gps']['last_known'] and re.search(r'no fresh location',part.text,re.I)):issues.append('unsupported_missing_location')
    if (data.physiology.hr_peak_bpm or data.physiology.hr_max_bpm) and re.search(r'(no |absence of |missing )(HRV (or |and ))?physiology( data| supplied| available)|physiology data (is |are )?(unavailable|missing)',part.text,re.I):issues.append('unsupported_missing_physiology')
    if data.button.state=='sos' and 'sos' not in part.text.lower():issues.append('sos_omitted')
    if name=='interpretation':
        if 'C' not in part.evidence_ids or len(part.evidence_ids)<2:issues.append('assessment_evidence_missing')
        if not set(part.evidence_ids)&{'P','M','A','V','B'}:issues.append('supporting_observation_evidence_missing')
        if re.search(r'\bevidence(?:\s+ids?)?\s*:',part.text,re.I):issues.append('evidence_ids_in_prose')
        if not re.search(r'\b(unconfirmed|unverified|unknown|unavailable|cannot|insufficient|inconclusive|possible|plausible|not|no)\b',part.text,re.I):issues.append('uncertainty_missing')
        if data.button.state=='sos':
            if not re.search(r'urgent SOS handling (?:remains |is )?(?:required|needed)',part.text,re.I):issues.append('sos_priority_missing')
            if re.search(r'(?:ignore|dismiss|cancel).{0,30}\bSOS\b|\bSOS\b[^.!?]{0,70}(?:unnecessary|not required|not needed|should be ignored)',part.text,re.I):issues.append('sos_priority_negated')
        if not data.audio.received and not re.search(r'missing audio|audio (is |was )?(not received|unavailable)|no audio|audio and speaker data (are |were )?unavailable',part.text,re.I):issues.append('missing_audio_limit_omitted')
        if view['gps']['last_known'] and not re.search(r'last.known|stale|current position.*unconfirmed',part.text,re.I):issues.append('location_limit_omitted')
    return issues


class IncidentReportService:
    def __init__(self,backend,*,mode=None):
        self.backend=backend
        self.mode=mode or os.getenv('WUALT_REPORT_MODE','guarded')
        if self.mode not in ('guarded','raw_review'):
            raise ValueError('WUALT_REPORT_MODE must be guarded or raw_review')

    def generate(self,data):
        from .contextual_report import generate_contextual_report
        return generate_contextual_report(data, self.backend, self.mode)
