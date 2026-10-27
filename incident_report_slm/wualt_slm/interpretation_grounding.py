"""Targeted semantic diagnostics; these are not a general truth verifier."""
import json
import re

CURRENT_TIMES = {"Within the supplied observation window", "At the snapshot time"}
REPORTING = r"\b(?:says?|said|reports?|reported|describes?|described|mentions?|mentioned|accounts?|according to|asks?|asked|requests?|denies?|denied|claims?|claimed)\b"
INJURY = r"\b(?:injury|injuries|injured|fracture[ds]?)\b"
MIXED = r"\b(?:mix of emotions|mixed (?:emotions|feelings)|conflicting (?:emotions|feelings)|ambivalen\w*|bittersweet)\b"
# A single comparison with a reference cannot show change or stability over time.
HEART_TREND = r"\b(?:remain(?:s|ed|ing)?|stay(?:s|ed|ing)?|steady|stable|stabl[ey]|throughout|over (?:the|this|that) (?:observation )?(?:window|period|time)|trend\w*|ris(?:e|es|ing)|rose|fell|fall(?:s|ing)|dropp?(?:ed|ing|s)?|increas\w*|decreas\w*|fluctuat\w*|consistent(?:ly)? (?:during|throughout|over|across))\b"
HEART = r"\b(?:heart|heartbeats?|pulse|readings?|measurements?)\b"
ACTOR = r"\b(?:my|the|his|her|their|our|a)\s+(character|avatar|hero|villain|player|doll|puppet)\b"
CALL = r"\b(?:shout\w*|yell\w*|scream\w*|call\w*|cr(?:y|ies|ied|ying)|ask\w*|beg\w*|plead\w*)\b"
FEAR = r"\b(?:scared|afraid|frightened|terrified|worried|panick\w*|upset|fear\w*)\b"
PAST_FEAR = r"\bI\s+(?:was|felt|had been|got)\s+(?:(?:so|very|really|quite|a bit|a little)\s+)?" + FEAR
CURRENT_OK = r"\b(?:I(?: am|'m)|I feel|I(?:'m| am) feeling|feel(?:ing)?)\s+(?:(?:much|a lot|fine|okay|ok|all)\s+)?(?:okay|ok|fine|calm|calmer|safe|better|alright|all right|relieved|good)\b|\b(?:okay|fine|calm|safe|better) now\b"
NEGATION = r"\b(?:not|no longer|never|doesn't|don't|didn't|cannot|can't|isn't|aren't|wasn't|hardly)\b"
SUPPORT = r"\b(?:align\w*|support\w*|consistent with|match\w*|in line with|corroborat\w*|reflect\w*|indicat\w*|suggest\w*|confirm\w*|explain\w*|establish\w*|show\w*)\b"
STATE_WORDS = r"\b(?:distress\w*|stress\w*|label\w*|state|classification|current)\b"


def label_support(text, *, negated):
    """Support verbs aimed at the state/label; negated means a negation directly precedes the verb."""
    found = []
    for match in re.finditer(SUPPORT, text, re.I):
        if not re.search(STATE_WORDS, text[match.end():match.end()+50], re.I):
            continue
        is_negated = bool(re.search(NEGATION + r"(?:\s+\w+){0,3}\s*$", text[max(0, match.start()-40):match.start()], re.I))
        if is_negated == negated:
            found.append(match)
    return found
# Anchored to the transcript: doubts about zero-confidence tone or sound labels are legitimate.
SPEECH_SOURCE = r"\b(?:transcript\w*|transcription|speech|spoken words?|recorded words)\b"
DOUBT = r"\b(?:not|unconfirmed|uncertain|unclear|unknown|limited|questionable|unverified|in doubt)\b"
RELIABILITY_DOUBT = (
    SPEECH_SOURCE + r"(?:'s)?[^.;]{0,30}\breliab\w*\b[^.;]{0,40}" + DOUBT
    + r"|\breliab\w*\s+of\s+(?:the\s+)?" + SPEECH_SOURCE + r"[^.;]{0,40}" + DOUBT
    + r"|\b(?:not|un)\w*\s+(?:explicitly\s+)?(?:confirmed|verified|established)\b[^.;]{0,20}\breliab\w*\s+of\s+(?:the\s+)?" + SPEECH_SOURCE
    + r"|\b(?:transcript|transcription|speech recognition)\b[^.;]{0,40}\b(?:unreliable|inaccurate|may be wrong|may be incomplete)\b"
    r"|\b(?:accuracy|quality)\s+of\s+the\s+(?:transcript|transcription)\b[^.;]{0,30}\b(?:uncertain|unclear|unknown|unconfirmed|not confirmed|limited)\b")


def affirmative_matches(pattern, text):
    """Exclude explicit negations/limits, not merely any nearby negative word."""
    for match in re.finditer(pattern, text, re.I):
        before = text[max(0, match.start()-100):match.start()]
        after = text[match.end():match.end()+65]
        if re.search(r"(?:\bno|\bnot|\bwithout)\s+(?:(?:a|an|any|confirmed|reported|physical|clear|evidence|of|signs|a diagnosis)\s+){0,5}$", before, re.I):
            continue
        # A negated verb may be followed by ordinary describing words ("a wrist", "a serious"),
        # but not by a new clause ("and has an").
        if re.search(r"(?:does not|doesn't|cannot|can't|do not|did not|is not|isn't|not)\s+(?:(?:yet|necessarily|clearly|by itself|alone)\s+)?(?:confirm|establish|prove|show|indicate|support|imply|suggest|mean|point to)\s+(?:(?!(?:and|but|or|has|have|had|is|was|are|were|with|which|that)\b)[\w'-]+\s+){0,3}$", before, re.I):
            continue
        if re.match(r"\s+(?:is|are|was|were)\s+(?:not|unconfirmed)|\s+(?:cannot|can't) be (?:confirmed|established)", after, re.I):
            continue
        yield match


ENGLISH_WORDS = r"\b(?:i|i'm|i am|me|my|the|a|an|is|am|are|was|were|and|but|to|of|in|on|it|you|not|no|please|help|can|this|that|have|has)\b"


def english_like(text):
    """Heuristic: the English-language checks can only read English observations.

    Non-Latin scripts and romanized Hindi/Tamil ("Mujhe darr lag raha", "Enakku romba
    tension") are treated as non-English, so English word patterns are not applied to them.
    """
    letters = re.findall(r"[^\W\d_]", text or "")
    if not letters:
        return True
    latin = sum(ch.isascii() for ch in letters) / len(letters)
    words = re.findall(r"[\w']+", (text or "").lower())
    english = len(re.findall(ENGLISH_WORDS, (text or "").lower()))
    return latin >= 0.9 and bool(words) and english / len(words) >= 0.2


def current_user_chat(available):
    return {key: item for key, item in available.items()
            if re.fullmatch(r"CHAT\d+", key) and item.get("role") == "user"
            and item.get("timing") in CURRENT_TIMES and item.get("usable", True)}


# The sentence denies support for a normal state while giving reasons that describe one.
CALM_REASON = r"\b(?:calm|relaxed|fine|okay|ok|cheerful|happy|just (?:playing|joking|singing|tired|waiting|resting)|not (?:in )?(?:distress\w*|stressed|scared|afraid|frightened|alarmed|worried|in danger)|no (?:stress|distress|fear|signs? of distress)|not the speaker'?s own|not (?:their|his|her) own)\b"
DANGER_CUE = r"\b(?:help|unsafe|danger\w*|threat\w*|hurt\w*|pain\w*|injur\w*|trapped|followed|attack\w*)\b"
CONFIDENCE_WORDS = r"\b(?:high|higher|very high|strong|full|good|great|considerable)\s+(?:level of\s+|degree of\s+)?(?:confidence|certainty)\b|\bconfident(?:ly)?\s+(?:detected|identified|classified)\b"
QUOTED = r"(?<!\w)['‘\"“]([^'’\"”]{2,60}?)['’\"”](?!\w)"


def inverted_normal(text):
    """'Does not support the normal state' whose stated reasons are calm/not distressed."""
    if not re.search(CALM_REASON, text, re.I):
        return False
    fears = [m for m in re.finditer(FEAR + "|" + DANGER_CUE, text, re.I)
             if not re.search(r"\b(?:not|no|never|without)\b(?:\s+\w+){0,2}\s*$", text[max(0, m.start()-25):m.start()], re.I)]
    if fears:
        return False
    return any(re.search(r"\bnormal\b", text[m.end():m.end()+50], re.I) for m in label_support(text, negated=True))


def copied_original_words(text, available):
    """Original-script letters, or quoted original-language words copied into the English paragraph."""
    if any(ch.isalpha() and ord(ch) > 0x24F for ch in text):
        return True
    chats = current_user_chat(available)
    keywords = available.get("KEYWORDS", {})
    sources = [available.get("TRANSCRIPT", {}).get("text") or ""] + [item["text"] for item in chats.values()] \
        + [w for key in ("stress", "distress") for w in keywords.get(key, [])]
    # Quoted words count as copied only when the speech or message itself is not English.
    observed = " ".join(s for s in sources[:1 + len(chats)] if s)
    if english_like(observed):
        return False
    foreign = [s.lower() for s in sources if s]
    for quoted in re.findall(QUOTED, text):
        phrase = quoted.strip(" .,!?;:").lower()
        if phrase and not english_like(phrase) and any(phrase in source for source in foreign):
            return True
    return False


def heart_trends(text):
    """Trend wording, excluding negations such as 'do not indicate a trend'."""
    return [m for m in re.finditer(HEART_TREND, text, re.I)
            if not re.search(r"\b(?:not|no|never|without|nor)\b(?:\s+[\w'-]+){0,3}\s*$", text[max(0, m.start()-40):m.start()], re.I)]


def _phrase(text, match, before=2):
    """The matched words plus a few preceding words, e.g. 'a wrist injury'."""
    head = re.findall(r"[\w'-]+", text[:match.start()])[-before:] if before else []
    return " ".join(head + [match.group(0).strip()]).strip(" ,.;:")


def revision_hints(draft, available, section):
    """Name the exact wording a rejected draft must change.

    Used only in the revision request sent to the model. These hints quote model prose and
    input words, so they are never stored in diagnostics or returned by the API.
    """
    hints = []
    text = " ".join(s.text for s in draft.sentences)
    speech = available.get("TRANSCRIPT", {}).get("text") or ""
    chats = current_user_chat(available)
    observation = " ".join([speech] + [item["text"] for item in chats.values()])
    def add(phrase, instruction):
        phrase = phrase.strip(" ,.;:")
        if phrase:
            hints.append("Replace " + json.dumps(phrase, ensure_ascii=False) + ": " + instruction)
    if not list(affirmative_matches(INJURY, observation)):
        for m in affirmative_matches(INJURY, text):
            add(_phrase(text, m), "the person reported pain or an event, not an injury; describe the pain instead.")
    for m in re.finditer(CONFIDENCE_WORDS, text, re.I):
        add(_phrase(text, m, 1), "describe the tone or sound itself without a confidence level.")
    for word in re.findall(r"\S*[^\x00-ɏ\s]\S*", text):
        # Whole non-Latin words: combining vowel signs are not word characters.
        add(word.strip("'\"()[]‘’“”.,;:!?"), "give its English meaning instead.")
    if not english_like(observation):
        for quoted in re.findall(QUOTED, text):
            phrase = quoted.strip(" .,!?;:")
            if phrase and not english_like(phrase):
                add(phrase, "give its English meaning instead of the original-language word.")
    if re.search(r"\bnearby\b", text, re.I) and not re.search(r"\bnearby\b", json.dumps(available), re.I):
        add("nearby", "sound detection gives no distance; say the sound was detected.")
    for sentence in draft.sentences:
        if set(sentence.evidence_ids) & {"HR", "HRV"} and re.search(HEART, sentence.text, re.I):
            for m in heart_trends(sentence.text):
                add(_phrase(sentence.text, m, 1), "a heart reading is one comparison with a reference, not a trend.")
    if section == "distress" and resolved_past_fear(available):
        for m in label_support(text, negated=False):
            add(_phrase(text, m, 2) + text[m.end():m.end()+40].split(".")[0], "past fear does not support current distress; say the current account does not clearly support it.")
    if section == "distress" and available.get("STATE", {}).get("label") == "normal":
        for sentence in draft.sentences:
            if inverted_normal(sentence.text):
                for m in label_support(sentence.text, negated=True):
                    add(_phrase(sentence.text, m, 3) + sentence.text[m.end():m.end()+25].split(",")[0], "the reasons describe a calm person; say the observations are consistent with the normal state.")
    if speech and available.get("TRANSCRIPT", {}).get("reliability") == "usable":
        for m in re.finditer(RELIABILITY_DOUBT, text, re.I):
            add(m.group(0), "the transcript is marked usable; remove this caveat.")
    return list(dict.fromkeys(hints))


def attributed_actor(speech):
    """The actor (e.g. a game character) the speech says is calling out, if any."""
    match = re.search(ACTOR + r"[^.!?]{0,60}" + CALL, speech or "", re.I)
    return match.group(1).lower() if match else None


def resolved_past_fear(available):
    """Fear described as past, plus a current statement of being okay or calm."""
    chats = current_user_chat(available)
    speech = available.get("TRANSCRIPT", {}).get("text") or ""
    current = " ".join([speech] + [item["text"] for item in chats.values()])
    return bool(re.search(PAST_FEAR, current, re.I) and re.search(CURRENT_OK, current, re.I))


def review_grounding_issues(draft, available, section):
    issues = []
    speech = available.get("TRANSCRIPT", {}).get("text") or ""
    reliability = available.get("TRANSCRIPT", {}).get("reliability")
    actor = attributed_actor(speech)
    resolved = resolved_past_fear(available)
    state = available.get("STATE", {}).get("label")
    chats = current_user_chat(available)
    all_text = " ".join(s.text for s in draft.sentences)
    cited = {key for s in draft.sentences for key in s.evidence_ids}
    observation_text = " ".join([speech] + [item["text"] for item in chats.values()])
    has_injury_account = bool(list(affirmative_matches(INJURY, observation_text)))
    # The injury check cannot read non-English observations. There, an injury may appear
    # only as the person's attributed account, never as an established fact.
    english_observations = english_like(observation_text)
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
    if any(not re.search(r"\b(?:not|never|without)\b[^.;]{0,25}$", all_text[max(0, m.start()-30):m.start()], re.I)
           for m in re.finditer(CONFIDENCE_WORDS, all_text, re.I)):
        issues.append("Do not describe tone or detection confidence as high, strong or moderate. Describe the observation itself; only a low-certainty sound may be qualified as possible or tentative.")
    if copied_original_words(all_text, available):
        issues.append("Write only in English. Give the English meaning instead of copying words from the original language or script, even in quotes or parentheses.")
    if speech and reliability == "usable" and re.search(RELIABILITY_DOUBT, all_text, re.I):
        issues.append("The transcript is marked usable. Do not add a transcription-reliability caveat; mention reliability only when it is limited, unreliable or unknown.")
    if section == "distress" and resolved and state in ("stress", "distress") and not (
            label_support(all_text, negated=True)
            or re.search(r"\b(?:conflict\w*|contrast\w*|contradict\w*|disagree\w*|inconsistent|at odds)\b[^.;]{0,50}" + STATE_WORDS, all_text, re.I)
            or re.search(r"\bno longer\b[^.;]{0,40}\b(?:distress\w*|stress\w*|afraid|scared|frightened)\b", all_text, re.I)):
        issues.append("The fear is described as past and the person now says they are okay or calm. Explain that this current account does not clearly support the supplied " + state + " state; do not leave the label as the conclusion.")
    for sentence in draft.sentences:
        text = sentence.text
        ids = set(sentence.evidence_ids)
        if section == "distress" and state == "normal" and inverted_normal(text):
            issues.append("This sentence says the observations do not support the normal state, but its reasons describe a calm or non-distressed person. If the observations fit the normal state, say they are consistent with it.")
        if ids & {"HR", "HRV"} and re.search(HEART, text, re.I) and heart_trends(text):
            issues.append("Each heart reading is one measurement compared with one reference. Do not describe readings as remaining, staying steady or consistent, or rising or falling over the window; state only the supplied comparison.")
        if actor and "TRANSCRIPT" in ids and re.search(CALL + r"[^.;]{0,25}\bhelp\b|\bhelp\b[^.;]{0,25}" + CALL, text, re.I) and actor not in text.lower():
            issues.append("The transcript says the " + actor + " is calling for help, not the person. Attribute the shout or request to the " + actor + "; do not say the person is shouting or asking for help.")
        if section == "distress" and resolved and re.search(FEAR, text, re.I) and label_support(text, negated=False):
            issues.append("Past fear is not current distress. Do not treat the fear word, an extracted keyword or the label as support for current distress when the person describes the fear as past and says they are okay or calm now.")
        injuries = [] if has_injury_account else list(affirmative_matches(INJURY, text))
        if injuries and not english_observations:
            injuries = [m for m in injuries if not re.search(REPORTING, text[:m.start()], re.I)]
        if injuries:
            issues.append("The observations report pain or an event, not a confirmed injury. Describe the reported symptom or event without upgrading it to an injury."
                          + ("" if english_observations else " If the original-language speech reports an injury, attribute it to the person (says/reports) instead of stating it as fact."))
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
