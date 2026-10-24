"""Bounded diagnostics for contextual reports; never a semantic certification."""
import re


def positive_sos_claim(text):
    for match in re.finditer(r'\bsos\b[^.!?]{0,35}\b(?:recorded|detected|triggered|pressed|activated)\b', text):
        prefix = text[max(0, match.start()-30):match.start()]
        if re.search(r'\b(?:no|without|not)\b[^.!?]*$', prefix) or re.search(r'\b(?:not|never)\b', match.group()):
            continue
        return True
    return False


def dismisses_sos(text):
    if re.search(r'\bsos\b[^.!?]{0,50}(?:unnecessary|not required|not needed|false alarm)', text):
        return True
    for match in re.finditer(r'\b(?:ignore|dismiss|cancel)\b[^.!?]{0,30}\bsos\b', text):
        prefix=text[max(0,match.start()-25):match.start()]
        if not re.search(r'\b(?:not|never|cannot|can.t|doesn.t|mustn.t)\b[^.!?]*$',prefix):
            return True
    return False


def contextual_issues(part, evidence, section):
    if not isinstance(part, dict) or not isinstance(part.get('text'), str) or not isinstance(part.get('evidence_ids'), list):
        return ['Return text and evidence_ids in the required JSON structure.']
    text = part['text']; ids = part['evidence_ids']; lower = text.lower()
    facts = {e['id']: e for e in evidence}
    if not ids or any(not isinstance(i, str) or i not in facts for i in ids) or len(ids) != len(set(ids)):
        return ['Cite supplied evidence IDs once each; do not invent citations.']
    cited = ' '.join(facts[i]['text'] for i in ids)
    issues = []
    if len(text.split()) > {'description': 100, 'interpretation': 170, 'clarification': 45}[section]:
        issues.append('Shorten this section without repeating facts or qualifications.')
    if re.search(r'\bevidence(?:\s+ids?)?\s*:|\b[HTE]\d{2}\b', text, re.I):
        issues.append('Keep evidence IDs outside the report prose.')
    if re.search(r'\b(?:backend|upstream|insufficient_context|override_marker|speaker_label|identity_verified|signal_quality|comparable_context|null)\b', lower):
        issues.append('Remove technical labels and instruction tokens from the prose; explain the incident itself.')
    if re.search(r'\b\d{1,2}:\d{2}(?::\d{2})?\b|\b20\d{2}-\d{2}-\d{2}\b', text):
        issues.append('Keep exact timestamps in the source record; describe relative sequence in prose.')
    if re.search(r'(?<![\w])\d+(?:\.\d+)?', text):
        numbers = {float(n) for n in re.findall(r'(?<![\w])\d+(?:\.\d+)?', text)}
        numeric_sources = cited + ' ' + ' '.join(facts[i].get('relative_time', '') for i in ids)
        if numbers - {float(n) for n in re.findall(r'(?<![\w])\d+(?:\.\d+)?', numeric_sources)}:
            issues.append('Use only numbers present in the cited facts; do not invent scores or measurements.')
    if re.search(r'\b(?:wearer|user|person) is safe\b|\bno danger\b|\b(?:confirms?|proves?) (?:an? )?(?:attack|assault|diagnosis|injury|emergency)\b', lower):
        issues.append('Remove unsupported confirmation of safety, diagnosis, injury or cause.')
    if re.search(r'heart.rate[^.!?]{0,100}(?:directly reflects|is caused by|proves|confirms)|\bnot (?:distress|an emergency)\b', lower):
        issues.append('Remove unsupported confirmation of a heart-rate cause or exclusion of distress; describe only a plausible explanation.')
    if re.search(r'\b(?:dispatch|responder|notification|alert|emergency response) (?:was |is |has been )?(?:sent|initiated|cancelled|completed|dispatched)\b', lower):
        issues.append('Do not invent operational actions or a response.')
    for observation in ('barking', 'screaming', 'raised voice', 'angry tone', 'irregular movement', 'heavy breathing'):
        if observation in lower and observation not in cited.lower():
            issues.append('Do not invent acoustic findings or movement detail from transcript text or an activity label.')
            break
    button = facts.get('B', {}).get('state')
    if button == 'sos':
        if dismisses_sos(lower):
            issues.append('Do not dismiss a recorded SOS or cancel its handling.')
        if section in ('description', 'interpretation') and ('sos' not in lower or 'B' not in ids):
            issues.append('Mention and cite the recorded SOS; contextual explanations do not cancel it.')
        if section == 'interpretation' and not re.search(r'(?:urgent|immediate)[^.!?]{0,35}(?:handling|attention|review|assistance|response)|(?:handling|attention|review|assistance|response)[^.!?]{0,35}(?:urgent|immediate)', lower):
            issues.append('Preserve urgent handling for the recorded device SOS.')
    elif positive_sos_claim(lower) or 'urgent sos handling' in lower:
        issues.append('Do not invent a recorded SOS or urgent SOS handling.')
    audio = facts.get('A', {})
    physiology = facts.get('P', {})
    if physiology.get('reference_type') == 'pre_event' and re.search(r'(?:resting|usual|normal) (?:heart.rate|baseline|reference)|(?:usual|resting) \d', lower):
        issues.append('Keep the supplied pre-event reference distinct from a resting, normal or usual baseline.')
    if facts.get('G', {}).get('available') and re.search(r'(?:no|absence of|lack of|without) (?:a )?(?:location|gps) (?:fix|data)', lower):
        issues.append('A location fix is supplied; its age does not mean location data is absent.')
    if not physiology.get('reference_available') and re.search(r'heart.rate[^.!?]{0,50}(?:above|below|increased from|higher than|lower than)[^.!?]{0,30}(?:reference|baseline)', lower):
        issues.append('No heart-rate reference was supplied; do not invent a baseline comparison.')
    if not physiology.get('reference_available') and re.search(r'(?:elevated|high|low|raised|increased|reduced) heart.rate|heart.rate[^.!?]{0,45}\b(?:is|was|is considered|which is) (?:elevated|high|low|raised|increased|reduced)\b', lower):
        issues.append('No heart-rate reference was supplied; do not classify the measurement as elevated, high or low.')
    if not physiology.get('available') and re.search(r'heart.rate (?:was |is )?(?:elevated|raised|high|low|increased)', lower):
        issues.append('No current heart-rate measurement was supplied; do not invent one.')
    if audio.get('available') and re.search(r'(?:no|missing|unavailable) transcript|transcript(?:ion)? (?:is |was )?(?:missing|unavailable|not supplied)', lower):
        issues.append('A transcript is supplied; distinguish unreliable text from missing text.')
    if re.search(r'audio (?:was |is )?not received|no audio was recorded|no recording (?:exists|was detected)', lower):
        issues.append('Transcript availability does not establish whether raw audio was received or recorded.')
    if physiology.get('available') and re.search(r'heart.rate(?! variability)|\bhr\b|\bbpm\b', lower) and 'P' not in ids:
        issues.append('Cite P for the heart-rate information used in this section.')
    if re.search(r'(?:normal|healthy|abnormal) (?:heart.rate|range|reading)|heart.rate[^.!?]{0,35}(?:normal|healthy|abnormal)', lower):
        issues.append('Remove unsupported clinical classification of the supplied heart-rate readings.')
    if section == 'interpretation' and not audio.get('available') and re.search(r'transcription reliability|unreliable transcript|limited transcript', lower):
        issues.append('No transcribed words are supplied; explain missing speech without assigning reliability to nonexistent wording.')
    if section == 'interpretation':
        if button != 'sos' and re.search(r'(?:no|absence of|lack of|without)[^.!?]{0,35}\bsos\b', lower):
            issues.append('Omit absent-SOS caveats; explain the available evidence instead.')
        if facts.get('K', {}).get('unmatched_keywords') and ('K' not in ids or not re.search(r'keyword|extraction|instruction',lower)):
            issues.append('Explain that unmatched keywords or instruction-like extraction hints do not establish events beyond the full transcript; cite K.')
        transcript_ids = {e['id'] for e in evidence if e['kind'] == 'transcript'}
        chat_ids = {e['id'] for e in evidence if e['kind'] == 'chat' and e.get('role') == 'user'}
        if transcript_ids and not transcript_ids.intersection(ids):
            issues.append('Use and cite the transcribed words, with their reliability, when explaining this incident.')
        if chat_ids and not chat_ids.intersection(ids):
            issues.append('Address and cite the user chat context, including when it is unrelated or outdated.')
        old_chat = any(e.get('role') == 'user' and e.get('seconds_before_incident', 0) >= 86400 for e in evidence)
        if old_chat and not re.search(r'\bdays?\b|outdated|historical|old (?:chat|message|plan)|stale', lower):
            issues.append('Explain that the historical user chat is days old and does not establish a current plan.')
        measurements = {e['id'] for e in evidence if e['kind'] in ('activity','measurement') and e.get('available')}
        if measurements and not transcript_ids and not chat_ids and not measurements.intersection(ids):
            issues.append('Connect the supplied activity or heart-rate observation to the contextual explanation.')
        if not re.search(r'could|may|might|suggest|consistent|possible|uncertain|unconfirmed|not confirmed|cannot|does not|not establish|unclear|unknown|insufficient', lower):
            issues.append('Qualify the interpretation and explain what the evidence cannot establish.')
        if audio.get('reliability') in ('limited', 'unreliable') and not re.search(r'transcri\w*[^.!?]{0,70}(?:limited|unreliable|uncertain|error)|(?:limited|unreliable|uncertain)[^.!?]{0,40}transcri', lower):
            issues.append('Acknowledge the supplied transcription reliability in the interpretation.')
        if len(text.split()) < 25:
            issues.append('Explain a specific relationship or conflict; a label alone is not an interpretation.')
    if section == 'clarification':
        if button == 'sos' and not re.search(r'help|assistan|need|safe|danger|emergency', lower):
            issues.append('For the recorded SOS, ask about assistance needed now, not routine travel plans.')
        if text.count('?') != 1:
            issues.append('Ask one concise incident-specific clarification question.')
        if re.search(r'\bexact (?:relationship|wording|cause|nature)\b|heart.rate|relationship between|consistent with[^?]{0,60}(?:label|activity)|\bhow.{0,65}\brelat', lower):
            issues.append('Ask the wearer about a concrete action or need for help, not to explain physiology or relationships between records.')
    return list(dict.fromkeys(issues))


def contextual_issue_rank(issues):
    severe = ('Do not invent', 'Remove unsupported', 'Do not dismiss', 'A transcript is supplied',
              'Transcript availability', 'Use only numbers', 'Mention and cite the recorded SOS', 'No heart-rate', 'Preserve urgent',
              'Do not compare HRV', 'Use the supplied HRV', 'Attribute unidentified',
              'Cite valid evidence IDs', 'Cite V for')
    # Presentation/coverage improvements must not buy a new detected factual or
    # SOS error, regardless of how many minor issues the revision removes.
    return sum(issue.startswith(severe) for issue in issues), len(issues)
