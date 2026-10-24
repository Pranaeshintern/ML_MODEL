"""Incident-v2 evidence and report assembly. No generated replacement prose."""
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from .backend import BackendUnavailable, RUNTIME_LOCK
from .incident_facts import build_incident_view
from .incident_schema import IncidentBundle, InterpretationDraft
from .contextual_quality import contextual_issues
from .physiology_evidence import physiology_facts
from .physiology_quality import sentence_issues

PROMPT_VERSION = 'wualt-interpretation-v11'


def build_contextual_view(data):
    # Reuse formatting and deterministic priority rules without pretending that
    # a transcript contains acoustic measurements. The original snapshot remains
    # unchanged in the output; this adapter is internal to legacy field rendering.
    legacy = data.model_dump(mode='python', exclude={'activity', 'chat_context'})
    legacy['physiology'].pop('hrv', None)
    legacy['physiology'].pop('other_measurements', None)
    legacy.update(schema_version='incident-v1',
                  motion={'observation': f'Activity classification: {data.activity.label}.' if data.activity else None},
                  audio={'received': bool(data.audio.transcript),
                         'reliability': data.audio.transcript_reliability})
    view = build_incident_view(IncidentBundle.model_validate(legacy))
    facts = []

    def add(key, text, kind, **metadata):
        facts.append({'id': key, 'text': text, 'kind': kind, **metadata})

    physiological = view['sensor_rows'][0]['observation']
    if data.physiology.hr_measured_at:
        physiological += ' Measured: ' + data.physiology.hr_measured_at.isoformat() + '.'
    if data.physiology.reference_at:
        physiological += ' Reference recorded: ' + data.physiology.reference_at.isoformat() + '.'
    peak = data.physiology.hr_peak_bpm or data.physiology.hr_max_bpm
    reference = data.physiology.reference_median_bpm
    comparison = ('above' if peak > reference else 'below' if peak < reference else 'equal') if peak and reference else 'unavailable'
    heart_facts = []
    if data.physiology.hr_min_bpm is not None:
        heart_facts.append(f'Recorded heart-rate range: {data.physiology.hr_min_bpm} to {data.physiology.hr_max_bpm} bpm.')
    if data.physiology.hr_peak_bpm is not None:
        heart_facts.append(f'Recorded peak: {data.physiology.hr_peak_bpm} bpm.')
    if reference is not None:
        heart_facts.append(f'Supplied {data.physiology.reference_type.replace("_", "-")} reference: {reference} bpm.')
    if not heart_facts:
        heart_facts.append('No heart-rate measurements supplied.')
    else:
        heart_facts.append(f'Heart-rate signal quality: {data.physiology.hr_signal_quality}.')
    add('P', ' '.join(heart_facts) or 'No heart-rate measurements supplied.', 'measurement',
        available=any(getattr(data.physiology, k) is not None for k in ('hr_peak_bpm', 'hr_max_bpm')),
        reference_available=reference is not None, peak_vs_reference=comparison,
        measured_peak_bpm=peak, reference_bpm=reference, reference_type=data.physiology.reference_type,
        signal_quality=data.physiology.hr_signal_quality,
        at=data.physiology.hr_measured_at.isoformat() if data.physiology.hr_measured_at else None)
    additional_physiology = physiology_facts(data)
    facts.extend(additional_physiology)
    activity = data.activity.label if data.activity else 'Not supplied'
    add('M', f'Activity label: {activity}.',
        'activity', available=data.activity is not None)
    reliability = data.audio.transcript_reliability
    add('A', f'Transcription reliability: {reliability}.',
        'transcription_status', reliability=reliability, available=bool(data.audio.transcript))
    transcript = []
    for index, line in enumerate(sorted(data.audio.transcript, key=lambda t: t.at), 1):
        speaker = f' [{line.speaker}]' if line.speaker else ''
        text = f'{line.at.isoformat()}{speaker}: {line.text}'
        transcript.append(text)
        add(f'T{index:02}', line.text, 'transcript', speaker_label=line.speaker,
            identity_verified=False, at=line.at.isoformat(), reliability=reliability,
            relative_time=(f'{abs((data.incident_start-line.at).total_seconds()):g} seconds ' + ('before' if line.at < data.incident_start else 'after' if line.at > data.incident_start else 'at') + ' incident start'))
    keywords = list(data.audio.keywords)
    if keywords:
        # Extraction hints share their source with the transcript. Matching a
        # substring is provenance bookkeeping, never a semantic assertion.
        matched = sorted({e['id'] for e in facts if e['kind'] == 'transcript'
                          and any(k.casefold() in e['text'].casefold() for k in keywords)})
        add('K', 'Supplied transcription keywords: ' + ', '.join(keywords) + '.',
            'derived_keywords', keywords=keywords, matched_transcript_ids=matched,
            unmatched_keywords=[k for k in keywords if not any(k.casefold() in line.text.casefold() for line in data.audio.transcript)],
            independent_evidence=False,
            limitation='Extraction hints only; retain negation and quotation in full transcript. Unmatched keywords are unverified.')
    chat = []
    for index, message in enumerate(sorted(data.chat_context.messages, key=lambda m: m.at), 1):
        relation = ('before the observation window' if message.at < data.observation_window.start
                    else 'after the observation window' if message.at > data.observation_window.end
                    else 'within the observation window')
        text = f'{message.at.isoformat()} [{message.role}; {relation}]: {message.text}'
        chat.append(text)
        seconds = (data.incident_start-message.at).total_seconds()
        age = abs(seconds)
        age_text = (f'{age/86400:.1f} days' if age >= 86400 else f'{age/3600:.1f} hours' if age >= 3600
                    else f'{age/60:.1f} minutes' if age >= 60 else f'{age:g} seconds')
        add(f'H{index:02}', message.text, 'chat', role=message.role, at=message.at.isoformat(),
            temporal_relation=relation, relative_time=age_text + (' before incident start' if seconds >= 0 else ' after incident start'),
            seconds_before_incident=seconds)
    add('HC', ('User-written chat messages are supplied.' if any(m.role == 'user' for m in data.chat_context.messages) else 'Only assistant messages are supplied; no user-written messages are available.' if data.chat_context.messages else 'No chat messages were supplied.'), 'chat_status', available=bool(data.chat_context.messages))
    for index, event in enumerate(sorted(data.events, key=lambda e: e.at), 1):
        add(f'E{index:02}', event.observation, 'recorded_event', at=event.at.isoformat())
    add('B', 'A device SOS button press is recorded.' if data.button.state == 'sos' else 'Device button state: ' + data.button.state + '.', 'button', state=data.button.state,
        at=data.button.at.isoformat() if data.button.at else None)
    location_text = ('Location fix supplied at ' + data.location.fixed_at.isoformat() + '. A recorded fix does not verify the current position.'
                     if data.location else 'No location fix supplied.')
    add('G', location_text, 'location', available=data.location is not None,
        last_known=bool(data.location and data.location.last_known),
        seconds_before_incident=(data.incident_start-data.location.fixed_at).total_seconds() if data.location else None)
    add('C', f'Backend-supplied assessment: {data.assessment.context}. This is context, not a required conclusion.', 'backend_context')
    limits = ['Cause and current safety are not established by these inputs.',
              'Transcribed speech is not verified as the wearer; speaker labels are identifiers only.',
              'Chat messages are statements, not independent observations. Assistant text is not user testimony.',
              'Activity labels do not establish intent or a detailed movement pattern.']
    if reliability == 'unavailable':
        limits.append('No transcription is available; this does not establish silence or absence of recording.')
    elif reliability != 'usable':
        limits.append(f'Transcription is {reliability}; exact wording may be wrong.')
    if not data.location:
        limits.append('No location was supplied.')
    elif data.location.last_known:
        limits.append('The last-known location does not establish the current position.')
    add('L', ' '.join(limits), 'limitations')
    for fact in facts:
        if fact.get('at') and data.button.at:
            delta = (datetime.fromisoformat(fact['at']) - data.button.at).total_seconds()
            fact['relative_to_button'] = ('at the recorded SOS time' if delta == 0 else f'{abs(delta):g} seconds ' + ('before' if delta < 0 else 'after') + ' the recorded SOS')
    view.update(evidence=facts, transcript='\n'.join(transcript) or 'Transcription unavailable.',
                chat_context='\n'.join(chat) or 'No chat context supplied.', activity=activity,
                transcript_reliability=reliability, limitations=limits,
                policy_notes=['Transcript text does not establish raw audio availability or acoustic findings.'],
                sensor_rows=[{'signal': 'Heart rate', 'observation': physiological + ' Signal quality: ' + data.physiology.hr_signal_quality + '.'},
                             *[{'signal': 'HRV' if f['id']=='V' else 'Additional signal', 'observation': f['record_text']} for f in additional_physiology],
                             {'signal': 'Activity label', 'observation': activity},
                             {'signal': 'Transcription reliability', 'observation': reliability},
                             {'signal': 'SOS button', 'observation': view['button']}])
    # Do not expose misleading legacy sound/voice conclusions or fallback text.
    for key in ('fallback_description', 'fallback_interpretation', 'speaker_note', 'speaker_count', 'gender_labels', 'keywords'):
        view.pop(key, None)
    view['keywords'] = keywords
    facts_by_id = {f['id']:f for f in facts}
    for row in view['sensor_rows']:
        if row['signal'] == 'Heart rate':
            row['interpretation'] = ('Peak is '+comparison+' the supplied reference; cause is not established.' if comparison != 'unavailable' else 'No supported reference comparison is available.')
        elif row['signal'] == 'HRV':
            hrv_fact = facts_by_id['V']
            row['interpretation'] = ('Comparable metric is '+hrv_fact['comparison']+' the supplied reference; this is not a diagnosis or emotional assessment.' if hrv_fact['comparison'] != 'unavailable' else 'Reference comparison unavailable: '+'; '.join(hrv_fact['comparison_limitations'])+'.')
        elif row['signal'] == 'Activity label':
            row['interpretation'] = 'Supplied activity classification; intent and cause are not established.'
        elif row['signal'] == 'Transcription reliability':
            row['interpretation'] = 'No words to interpret.' if not data.audio.transcript else 'Reliability applies to transcription; speaker identity is unverified.'
        elif row['signal'] == 'SOS button':
            row['interpretation'] = 'Recorded device SOS; urgent handling remains required.' if data.button.state == 'sos' else 'Button status alone does not establish safety.'
        else:
            row['interpretation'] = 'Supplied measurement and quality only; no clinical classification is assigned.'
    return view


def generate_contextual_report(data, backend, mode):
    original = data
    data = contextual_input(data)
    view = build_contextual_view(data)
    if original.schema_version == 'incident-v1':
        add_legacy_observations(view, original)
    try:
        draft = InterpretationDraft.model_validate_json(backend.generate_contextual_incident(view['evidence']))
    except ValidationError as exc:
        raise BackendUnavailable('invalid_contextual_narrative_schema') from exc
    narratives = {'interpretation': draft.interpretation.model_dump()}
    supplied = {'description': original.incident_description, 'clarification': original.clarification_required}
    issues = [f'{name}:{issue}' for name, part in narratives.items()
              for issue in contextual_issues(part, view['evidence'], name)]
    issues.extend('interpretation:' + issue for sentence in draft.interpretation_sentences
                  for issue in sentence_issues(sentence.model_dump(), view['evidence']))
    issues = list(dict.fromkeys(issues))
    if mode == 'guarded' and issues:
        raise BackendUnavailable('contextual_quality_rejected')
    narratives.update({name: {'text': value or 'Not supplied.', 'evidence_ids': [], 'source': 'supplied' if value else 'unavailable'} for name, value in supplied.items()})
    return {'id': str(uuid4()), 'status': 'unvalidated_draft' if mode == 'raw_review' else 'draft',
            'input': original.model_dump(mode='json'), 'template_version': 'interpretation-only-v1', 'view': view, 'narrative': narratives,
            'fallback_enabled': False, 'report_mode': mode,
            'generation_mode': 'model_unvalidated' if mode == 'raw_review' else 'model',
            'model_sections': ['interpretation'], 'template_sections': ['description', 'clarification'], 'validation_issues': issues,
            'interpretation_attempts': draft.interpretation_attempts,
            'interpretation_selected_attempt': draft.interpretation_selected_attempt,
            'interpretation_sentences': [s.model_dump() for s in draft.interpretation_sentences],
            'interpretation_attempts_trace': [a.model_dump() for a in draft.interpretation_attempts_trace],
            'prompt_version': PROMPT_VERSION, 'model_id': RUNTIME_LOCK['model_id'],
            'model_revision': RUNTIME_LOCK['model_revision'],
            'generated_at': datetime.now(timezone.utc).isoformat()}


def contextual_input(data):
    """Adapt legacy observations without changing the recorded input or trigger."""
    if data.schema_version == 'incident-v2':
        return data
    raw = data.model_dump(mode='json')
    raw.update(schema_version='incident-v2', motion={}, audio={
        'transcript': raw['audio']['transcript'], 'keywords': raw['audio']['keywords'],
        'transcript_reliability': (raw['audio']['reliability'] if raw['audio']['reliability'] != 'unavailable' else 'unreliable') if raw['audio']['transcript'] else 'unavailable'})
    return IncidentBundle.model_validate(raw)


def add_legacy_observations(view, original):
    for key, label, value in [('M', 'Reported motion', original.motion.observation),
                              ('S', 'Supplied sounds', original.audio.sounds),
                              ('O', 'Supplied tone', original.audio.tone)]:
        if value:
            view['evidence'] = [f for f in view['evidence'] if f['id'] != key]
            view['evidence'].append({'id': key, 'kind': 'reported_observation', 'text': value,
                                     'available': True, 'reliability': original.audio.reliability if key != 'M' else 'reported'})
            view['sensor_rows'].append({'signal': label, 'observation': value})
