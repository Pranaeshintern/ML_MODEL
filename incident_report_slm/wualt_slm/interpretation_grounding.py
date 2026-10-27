"""Targeted semantic diagnostics; these are not a general truth verifier."""
import re

CURRENT_TIMES = {"Within the supplied observation window", "At the snapshot time"}
REPORTING = r"\b(?:says?|said|reports?|reported|describes?|described|mentions?|mentioned|accounts?|according to|asks?|asked|requests?|denies?|denied|claims?|claimed)\b"
INJURY = r"\b(?:injury|injuries|injured|fracture[ds]?)\b"
MIXED = r"\b(?:mix of emotions|mixed (?:emotions|feelings)|conflicting (?:emotions|feelings)|ambivalen\w*|bittersweet)\b"


def affirmative_matches(pattern, text):
    """Exclude explicit negations/limits, not merely any nearby negative word."""
    for match in re.finditer(pattern, text, re.I):
        before = text[max(0, match.start()-100):match.start()]
        after = text[match.end():match.end()+65]
        if re.search(r"(?:\bno|\bnot|\bwithout)\s+(?:(?:a|an|any|confirmed|reported|physical|clear|evidence|of|signs|a diagnosis)\s+){0,5}$", before, re.I):
            continue
        if re.search(r"(?:does not|doesn't|cannot|can't|do not|did not)\s+(?:confirm|establish|prove|show|indicate|support|imply)\s+(?:(?:a|an|any|the|physical|confirmed)\s+){0,3}$", before, re.I):
            continue
        if re.match(r"\s+(?:is|are|was|were)\s+(?:not|unconfirmed)|\s+(?:cannot|can't) be (?:confirmed|established)", after, re.I):
            continue
        yield match


def current_user_chat(available):
    return {key: item for key, item in available.items()
            if re.fullmatch(r"CHAT\d+", key) and item.get("role") == "user"
            and item.get("timing") in CURRENT_TIMES and item.get("usable", True)}


def review_grounding_issues(draft, available, section):
    issues = []
    speech = available.get("TRANSCRIPT", {}).get("text") or ""
    chats = current_user_chat(available)
    all_text = " ".join(s.text for s in draft.sentences)
    cited = {key for s in draft.sentences for key in s.evidence_ids}
    observation_text = " ".join([speech] + [item["text"] for item in chats.values()])
    has_injury_account = bool(list(affirmative_matches(INJURY, observation_text)))
    # Crying/tears and an upstream distress label alone cannot contradict explicit joy.
    explicit_joy = re.search(r"\b(?:happy|happiness|joy|joyful|delighted)\b", speech, re.I)
    contrary_affect = re.search(r"\b(?:sad|sadness|afraid|scared|frightened|unsafe|angry|upset|worried|grief|anxious|mixed|bittersweet)\b", observation_text, re.I)
    tone = available.get("TONE", {})
    if tone.get("usable", True) and tone.get("confidence") != 0:
        contrary_affect = contrary_affect or re.search(r"\b(?:sad|frightened|angry|worried|anxious|fearful)\b", tone.get("label", ""), re.I)
    if explicit_joy and not contrary_affect and list(affirmative_matches(MIXED, all_text)):
        issues.append("Preserve the explicitly happy meaning. Crying and the supplied state do not establish mixed emotions; do not add conflicting feelings absent from the observations.")
    if section == "distress" and chats and not (cited & chats.keys()):
        issues.append("Consider the current user chat as written testimony and cite its source. Do not silently discard it when assessing the supplied state; distinguish it from spoken audio.")
    for sentence in draft.sentences:
        text = sentence.text
        ids = set(sentence.evidence_ids)
        if not has_injury_account and list(affirmative_matches(INJURY, text)):
            issues.append("The observations report pain or an event, not a confirmed injury. Describe the reported symptom or event without upgrading it to an injury.")
        # Known external-event assertions require account attribution, even for a verified identity.
        external = re.search(r"\b(?:is|are|was|were) (?:being )?(?:followed|chased|threatened|trapped|attacked)\b|\b(?:person|speaker|father|mother|friend|child) (?:fell|slipped|crashed)\b", text, re.I)
        if "TRANSCRIPT" in ids and external and not re.search(REPORTING, text[:external.start()], re.I):
            issues.append("Attribute the external event to the speaker's account (says/reports/describes); identity verification does not independently confirm that it happened.")
        for key in ids:
            item = available.get(key, {})
            confidence = item.get("confidence")
            if re.fullmatch(r"EVENT\d+", key) and isinstance(confidence, (int, float)) and 0 < confidence < 0.5:
                # Uncertain identity/source is not uncertainty about detection itself.
                event_words = [re.escape(w) for w in re.findall(r"[a-z]+", item.get("label", "").lower()) if len(w) > 2]
                sound = r"(?:sound|noise|detection|detect(?:ed|ion)|scream|gunshot|barking|" + "|".join(event_words or ["event"]) + r")"
                if not re.search(r"\b" + sound + r"\b", text, re.I):
                    # A conclusion may cite the event without asserting detection again.
                    continue
                qualified = re.search(
                    r"\b(?:possible|possibly|potential|tentative|tentatively|uncertain|unclear|unreliable|unconfirmed|low.confidence)\s+(?:[\w-]+\s+){0,3}" + sound + r"\b"
                    + r"|\b(?:detection|sound|noise|label)(?:'s)?(?: nature| type)?\s+(?:is|was|remains|seems)\s+(?:uncertain|unclear|unreliable|unconfirmed)\b"
                    + r"|\b(?:detected|identified|detection)(?: for (?:a |an )?\w+)?\s+(?:with |at |has |had )?(?:very )?low (?:confidence|certainty)\b"
                    + r"|\b(?:may|might|could) (?:have been|be)\s+(?:a |an )?" + sound + r"\b"
                    + r"|\b[\w-]+-like (?:sound|noise)\b", text, re.I)
                if not qualified:
                    issues.append("The sound detection itself has low certainty. Say a possible/tentatively detected sound or that detection is uncertain; missing speech or unverified identity does not express this limit.")
        if section == "distress" and ids & chats.keys():
            written = re.search(r"\b(?:chat|message[sd]?|writ(?:es|ten|ing)|wrote|texts?|texted)\b", text, re.I)
            spoken = re.search(r"\b(?:audio|spoken|speech|aloud|verbally|says?|said)\b", text, re.I)
            written_context = written or (not speech and re.search(r"\b(?:chat|messages?|written|wrote|texted)\b", all_text, re.I))
            if not written_context or ("TRANSCRIPT" in ids and speech and not spoken):
                issues.append("Identify the channels: replace says/reports for written chat content with an explicit wrote/message/chat attribution. Attribute the spoken audio separately. Do not merge both channels into one spoken account or treat either as independently verified truth.")
    return issues
