"""Focused consistency checks used for one bounded model revision.

These detect specific omissions/contradictions, not arbitrary semantic errors.
They never supply replacement prose. Raw review retains the selected model text without rewriting it.
"""
import re


def interpretation_issues(part, evidence):
    if not isinstance(part,dict) or not isinstance(part.get('text'),str) or not isinstance(part.get('evidence_ids'),list):
        return ['Return a JSON object containing text and evidence_ids.']
    text=part['text'];lower=text.lower();ids=part['evidence_ids'];facts={f['id']:f['text'] for f in evidence}
    issues=[]
    if any(not isinstance(i,str) or i not in facts for i in ids) or len(ids)!=len(set(str(i) for i in ids)):
        issues.append('Cite only supplied evidence IDs, each once.')
    if not ids or ids[0]!='C':issues.append('Start evidence_ids with C for the supplied assessment.')
    if 'L' not in ids:issues.append('Cite L and state the relevant uncertainty.')
    assessment=facts.get('C','').split('. ')[0].rstrip('.')
    if assessment and not text.startswith(assessment+'.'):issues.append('Start with the exact supplied assessment sentence: '+assessment+'.')
    sos=facts.get('B','').startswith('A device SOS was recorded.')
    if sos:
        if 'urgent sos handling remains required.' not in lower or 'B' not in ids:
            issues.append('Include the sentence Urgent SOS handling remains required. Cite B.')
    elif re.search(r'\bsos\b|urgent handling',lower):issues.append('No recorded SOS is established. Remove all SOS and urgent-handling claims; do not infer button status.')
    if re.search(r'\bevidence(?:\s+ids?)?\s*:',lower):issues.append('Keep citation IDs out of the prose.')
    if len(text.split())>110:issues.append('Keep the paragraph under 110 words without repeating findings.')
    received=facts.get('A','').startswith('Audio was received')
    if received and re.search(r'(?:missing|no|absent) audio|audio (?:is |was )?(?:unavailable|not received|absent)',lower):
        issues.append('Audio was received. Do not say it is missing or unavailable.')
    if not received and not re.search(r'audio[^.!?]{0,70}(?:unavailable|not received|absent)|(?:missing|no) audio',lower):
        issues.append('Explicitly state that audio is unavailable.')
    if received and any(q in facts['A'] for q in ('limited','unreliable')) and not re.search(r'limited|unreliable|uncertain',lower):
        issues.append('Explain that received audio is limited or unreliable, as stated in A.')
    hr=not facts.get('P','Heart-rate measurements are not supplied.').startswith('Heart-rate measurements are not supplied.')
    if hr and re.search(r'no (?:hrv or (?:other )?)?(?:physiology|heart.rate)|(?:heart.rate|physiology)[^.!?]{0,20}(?:unavailable|not supplied|absent)',lower):
        issues.append('Heart-rate measurements are supplied. Do not claim they are absent.')
    location=facts.get('G','')
    if location.startswith('A location fix') and re.search(r'last.known|location[^.!?]{0,20}(?:not provided|not supplied|unavailable)|no location',lower):
        issues.append('Describe G only as: A location fix was supplied.')
    if location.startswith('Last-known') and not re.search(r'current position[^.!?]{0,30}(?:unconfirmed|unknown|not (?:confirmed|established))|(?:not|cannot)[^.!?]{0,30}(?:establish|confirm)[^.!?]{0,30}current position',lower):
        issues.append('State that the last-known fix leaves current position unconfirmed.')
    if location.startswith('No location') and re.search(r'last.known|location fix was supplied',lower):issues.append('No location fix is supplied; do not invent one.')
    available=[]
    if not facts.get('M','Motion data not supplied.').startswith('Motion data not supplied'):available.append(('M',r'motion|movement|running|cadence|walking'))
    if received:available.append(('A',r'audio|sound|bark|breathing|scream'))
    if hr:available.append(('P',r'heart.rate'))
    described=sum(bool(re.search(pattern,lower)) and key in ids for key,pattern in available)
    if described<min(2,len(available)):
        issues.append('Describe and cite at least two available sensor observations (motion, sound/audio, heart rate); do not merely list citation IDs or the SOS.')
    if not received and re.search(r'no (?:observed )?(?:movement or sound|sound or movement)|silence was',lower):
        issues.append('Missing data does not prove silence or absence of movement; remove that inference.')
    return issues


def issue_rank(issues):
    """Prefer fewer detected factual/coverage errors, then fewer format errors.

    A revision is another model candidate, not necessarily an improvement.
    Equal scores keep the first candidate. This does not rank unknown errors.
    """
    presentation=('Cite only supplied evidence IDs', 'Start evidence_ids',
                  'Cite L and', 'Keep citation IDs', 'Keep the paragraph')
    substantive=sum(not issue.startswith(presentation) for issue in issues)
    return substantive,len(issues)
