"""Explicit simulated responses for contract/renderer tests, not live accuracy."""
import json
from copy import deepcopy

def draft(part):
    part = deepcopy(part)
    return {'interpretation': part, 'interpretation_sentences': [dict(part, kind='interpretation')]}

class FixtureModel:
    def generate_contextual_incident(self, evidence):
        ids = [e['id'] for e in evidence][:25]
        text = 'The transcript contains a request for help with a dog; the motion report and heart-rate comparison may provide context but do not establish the cause. Transcription is limited and unmatched keywords do not establish additional events. The recorded SOS requires urgent handling.'
        return json.dumps(draft({'text': text, 'evidence_ids': ids}))
