"""Evidence-local diagnostics, not a medical or semantic correctness judge."""
import re
from datetime import datetime


def sentence_issues(sentence, evidence):
    facts = {e['id']: e for e in evidence}
    ids = sentence['evidence_ids']
    text = sentence['text'].lower()
    if not ids or len(ids) != len(set(ids)) or any(i not in facts for i in ids):
        return ['Cite valid evidence IDs for each individual interpretation sentence.']
    issues = []
    cited = ' '.join(facts[i]['text'] + ' ' + facts[i].get('relative_time', '') for i in ids)
    numbers = lambda value: {float(n) for n in re.findall(r'(?<![\w])\d+(?:\.\d+)?', value)}
    if numbers(text) - numbers(cited):
        issues.append('Use only numbers supported by this sentence’s own citations.')
    if re.search(r'\bhrv\b|\brmssd\b|\bsdnn\b|heart.rate variability', text):
        if re.search(r'parasympathetic|sympathetic|vagal|autonomic|physiological (?:stress|arousal)', text):
            issues.append('Remove unsupported autonomic or stress interpretation; report only the supplied HRV comparison and its limits.')
        if 'V' not in ids:
            issues.append('Cite V for this sentence’s HRV claim.')
        hrv = facts.get('V', {})
        if not hrv:
            issues.append('Do not invent an HRV measurement.')
        if hrv.get('comparison') == 'unavailable' and re.search(r'(?:hrv|rmssd|sdnn|heart.rate variability)[^;.!?]{0,55}\b(?:lower|higher|below|above|reduced|decreased|increased)\b|\b(?:lower|higher|reduced|decreased|increased) (?:hrv|rmssd|sdnn)\b', text):
            issues.append('Do not compare HRV with this reference; its measurement context is incompatible or unverified.')
        comparison = hrv.get('comparison')
        reverse = {'above': r'below|lower|decreased|reduced', 'below': r'above|higher|increased',
                   'equal': r'below|lower|decreased|reduced|above|higher|increased'}.get(comparison)
        if reverse and re.search(r'(?:hrv|rmssd|sdnn|heart.rate variability)[^.!?]{0,35}\b(?:' + reverse + r')\b', text):
            issues.append('Use the supplied HRV comparison direction; do not reverse it or invent a change.')
        if re.search(r'(?:low|high) (?:hrv|rmssd|sdnn)|(?:hrv|rmssd|sdnn) (?:is|was) (?:low|high)\b', text):
            issues.append('Describe HRV only relative to a compatible supplied reference, without a clinical high/low label.')
    if re.search(r'\b(?:stress|fear|panic|anxiety|distress)\b', text) and re.search(r'\bhrv\b|\brmssd\b|\bsdnn\b|heart.rate', text):
        if not re.search(r'cannot|does not|do not|not establish|not prove|not confirm|not identify|not determine|unconfirmed|unknown', text):
            issues.append('Remove unsupported emotional or causal interpretation of physiology.')
        for match in re.finditer(r'\b(?:suggests?|indicates?|reflects?|consistent with)\b[^.!?]{0,55}\b(?:stress|fear|panic|anxiety)\b', text):
            if not re.search(r'not|cannot|never', text[max(0,match.start()-25):match.start()]):
                issues.append('Remove unsupported emotional interpretation from physiological comparisons, even when phrased as a possibility.')
    if re.search(r'\b(?:denial|denies|denying)\b', text) and re.search(r'\b(?:hrv|heart.rate|physiolog\w*)\b', text) and re.search(r'contradict|discrepancy|inconsisten|conflict', text):
        if not re.search(r'does not contradict|not (?:a |an )?(?:conflict|discrepancy|inconsistency)', text):
            issues.append('Remove unsupported conflict between a denial of distress and physiological readings.')
    if re.search(r'dizz', text) and re.search(r'stationary|sitting|seated', text) and re.search(r'does not align|contradict|incompatible', text):
        issues.append('Remove unsupported conflict: sitting or being stationary is compatible with a reported symptom.')
    if any(facts[i]['kind'] == 'transcript' for i in ids) and not any(facts[i].get('role') == 'user' for i in ids):
        if re.search(r'(?:the )?(?:user|wearer) (?:said|says|stated|mentioned|reported|asked|described|denied|explained)', text):
            issues.append('Attribute unidentified transcribed speech to the speaker, not the wearer or user.')
    if ids == ['K'] and re.search(r'confirms?|corroborates?|proves?|establishes?', text) and not re.search(r'not|cannot', text):
        issues.append('Keywords are derived hints, not independent confirmation.')
    button = facts.get('B', {})
    if button.get('state') == 'sos' and button.get('at'):
        deltas = [(datetime.fromisoformat(facts[i]['at']) - datetime.fromisoformat(button['at'])).total_seconds()
                  for i in ids if facts[i]['kind'] in ('transcript', 'chat') and facts[i].get('at')]
        after = re.search(r'\bafter\b[^.!?]{0,35}\b(?:sos|button|activation|press)\b', text)
        before = re.search(r'\bbefore\b[^.!?]{0,35}\b(?:sos|button|activation|press)\b', text)
        if deltas and ((after and all(d < 0 for d in deltas)) or (before and all(d > 0 for d in deltas))):
            issues.append('Use the supplied speech/chat timing relative to the recorded SOS; do not reverse before and after.')
        if re.search(r'\b(?:wearer|user) (?:activated|pressed|triggered)\b[^.!?]{0,35}\b(?:sos|button)\b', text):
            issues.append('Describe a recorded device SOS without inventing who operated the button.')
    return issues
