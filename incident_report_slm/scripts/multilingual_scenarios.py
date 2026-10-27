"""Fictional multilingual transcripts and messages; never generated interpretations.

The wrapper receives transcripts, not audio. These cases test whether meaning, negation,
tense and attribution survive when the speech or message is not in English, including
non-Latin scripts, romanized speech and English mixed with another language.
"""
import copy
from datetime import datetime, timedelta


def multilingual_cases():
    rows = []

    def add(key, title, group, language, speech, meaning, tone, state, focus, *, sounds=(), keywords=(),
            physiology=None, activity=None, chat=(), verified=False, reliability="usable"):
        end = datetime.fromisoformat("2026-10-28T09:00:00+05:30") + timedelta(minutes=3 * len(rows))
        stamp = lambda seconds=0: (end + timedelta(seconds=seconds)).isoformat()
        payload = {"schema_version": "interpretation-v1", "request_id": "multilingual-" + key,
                   "time": stamp(), "observation_window": {"start": stamp(-60), "end": stamp()},
                   "state": {"label": state, "confidence": 0.82}, "transcript": speech,
                   "transcript_reliability": reliability if speech else "unavailable", "speaker_verified": verified,
                   "tone": {"label": tone, "confidence": 0.85} if tone else None,
                   "keywords": {"stress": list(keywords) if state == "stress" else [], "distress": list(keywords) if state != "stress" else []},
                   "events": [{"label": label, "confidence": confidence, "at": stamp(-20)} for label, confidence in sounds],
                   "hr": None, "hrv": None, "location": None, "activity": activity,
                   "chat_context": [{"role": role, "text": text, "at": stamp(offset)} for role, text, offset in chat]}
        if physiology:
            bpm, variation = physiology
            payload["hr"] = {"bpm": bpm, "at": stamp(-15), "signal_quality": "usable", "reference": {"bpm": 76, "at": stamp(-600), "comparable_context": True}}
            payload["hrv"] = {"metric": "rmssd", "value_ms": variation, "window_start": stamp(-60), "window_end": stamp(), "signal_quality": "usable", "artifact_corrected": True,
                              "reference": {"metric": "rmssd", "value_ms": 40, "window_seconds": 60, "at": stamp(-600), "comparable_context": True}}
        described = ("Speech supplied (" + language + "): " + speech + " Meaning: " + meaning) if speech else ("Written message (" + language + "). Meaning: " + meaning)
        rows.append({"scenario": key, "title": title, "category": group, "language": language,
                     "description": described, "review_focus": focus, "input": copy.deepcopy(payload)})

    raised = (118, 23)
    g = "Reviewed failure types in other languages"
    add("hindi-past-fear", "Past fear, now okay (Hindi)", g, "Hindi, Devanagari",
        "पिछले हफ्ते बिजली जाने पर मैं डर गया था। आज मैं ठीक हूँ।",
        "I got scared last week when the power went out. Today I am fine.",
        "reflective", "distress", "Past fear and 'fine today' must not support current distress.", keywords=["डर"], physiology=raised)
    add("tanglish-past-fear", "Past fear, now okay (romanized Tamil and English)", g, "Tamil-English, romanized",
        "Last week current pogumbodhu enakku bayama irundhuchu. Ippo naan okay.",
        "I was scared last week when the power went. Now I am okay.",
        "reflective", "distress", "Past fear and 'okay now' must not support current distress.", keywords=["bayama"])
    add("tamil-game-character", "A game character shouts for help (Tamil)", g, "Tamil",
        "என் கேரக்டர் கேமில் உதவி என்று கத்துகிறது. நான் சும்மா விளையாடுகிறேன்.",
        "My character is shouting help in the game. I am just playing.",
        "relaxed", "normal", "The call for help belongs to the game character, not the person.", keywords=["உதவி"])
    add("hinglish-game-character", "A game character shouts for help (romanized Hindi and English)", g, "Hindi-English, romanized",
        "Mera character game mein help help chilla raha hai. Main bas khel raha hoon.",
        "My character is shouting help in the game. I am just playing.",
        "relaxed", "normal", "The call for help belongs to the game character, not the person.", keywords=["help"])
    add("hindi-injury-report", "An explicit injury report (Hindi)", g, "Hindi, Devanagari",
        "मैं सीढ़ियों से गिर गया और मेरी कलाई टूट गई है। मुझे मदद चाहिए।",
        "I fell down the stairs and my wrist is broken. I need help.",
        "pained", "distress", "The person reports a broken wrist; attribute it to their account without diagnosing.", verified=True)
    add("telugu-pain-only", "Pain without an injury report (Telugu)", g, "Telugu",
        "నా మోకాలు చాలా నొప్పిగా ఉంది. నేను నడవలేకపోతున్నాను.",
        "My knee hurts a lot. I cannot walk.",
        "pained", "distress", "Pain and difficulty walking must not be upgraded to an injury.")
    add("french-chat-calm", "Spoken past fear and a current calm message (French)", g, "French",
        "J'avais peur quand l'ascenseur s'est arrêté.",
        "I was scared when the lift stopped. Message: the lift is working again and I am calm now.",
        "reflective", "distress", "Keep spoken and written channels apart; weigh the calm message against the label.",
        chat=[("user", "L'ascenseur remarche, je suis calme maintenant.", -10)])

    g = "Meaning and attribution across languages"
    add("spanish-sister-fear", "A sister's fear, not the speaker's (Spanish)", g, "Spanish",
        "Mi hermana tiene mucho miedo porque se fue la luz. Yo estoy tranquilo.",
        "My sister is very scared because the power went out. I am calm.",
        "calm", "distress", "The fear belongs to the sister; the speaker says they are calm.", keywords=["miedo"])
    add("hinglish-denial", "A denial of being followed (romanized Hindi)", g, "Hindi, romanized",
        "Koi mera peecha nahi kar raha. Main bas cab ka wait kar raha hoon.",
        "Nobody is following me. I am just waiting for a cab.",
        "calm", "normal", "Preserve the denial; do not treat 'following' as a report of pursuit.", keywords=["peecha"])
    add("marathi-negation", "Not scared, just tired (Marathi)", g, "Marathi",
        "मला भीती वाटत नाही. मी फक्त थकलो आहे.",
        "I am not scared. I am just tired.",
        "tired", "stress", "Preserve the negation; tiredness is not fear.", keywords=["भीती"])
    add("malayalam-joy-tears", "Crying with happiness (Malayalam)", g, "Malayalam",
        "സന്തോഷം കൊണ്ടാണ് ഞാൻ കരയുന്നത്.",
        "I am crying because of happiness.",
        "crying", "distress", "Keep the happy reason; do not add mixed emotions or distress.")
    add("arabic-song-quote", "Singing a song that says help me (Arabic)", g, "Arabic",
        "أنا أغني أغنية تقول: ساعدوني.",
        "I am singing a song that says: help me.",
        "cheerful", "normal", "The words 'help me' are song lyrics, not a request.", keywords=["ساعدوني"])

    g = "Requests for help across languages"
    add("bengali-followed", "Fear of being followed (Bengali)", g, "Bengali",
        "আমার খুব ভয় করছে। কেউ আমার পিছু নিচ্ছে। দয়া করে সাহায্য করুন।",
        "I am very scared. Someone is following me. Please help.",
        "frightened", "distress", "Attribute the pursuit to the person's report; do not confirm it.", physiology=raised, activity="walking")
    add("chinese-knocking", "Help with knocking at the door (Chinese)", g, "Chinese, simplified",
        "救命！有人在敲我的门，我很害怕。",
        "Help! Someone is knocking on my door, I am very scared.",
        "frightened", "distress", "Knocking does not establish a break-in; keep the fear and request.", sounds=[("knocking", 0.9)])
    add("kannada-normal", "Going home, all fine (Kannada)", g, "Kannada",
        "ನಾನು ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ. ಎಲ್ಲವೂ ಚೆನ್ನಾಗಿದೆ.",
        "I am going home. Everything is fine.",
        "calm", "normal", "Keep the ordinary meaning without certifying safety.", activity="walking")
    add("german-exercise", "Out of breath after running (German)", g, "German",
        "Ich bin gerade fünf Kilometer gelaufen und bin außer Atem.",
        "I just ran five kilometres and I am out of breath.",
        "breathless", "stress", "Exertion explains breathlessness and heart readings; it is not emotional stress.", physiology=raised, activity="running")
    add("hindi-limited-transcript", "A help request with limited transcript quality (Hindi)", g, "Hindi, Devanagari",
        "मदद... दरवाज़ा... बंद...", "Help... door... closed...",
        "frightened", "distress", "Limited reliability: do not build a full story from fragments.", reliability="limited")
    add("spanish-chat-only", "A written request for help only (Spanish)", g, "Spanish",
        None, "I am locked in the bathroom and cannot get out. Please call my mother.",
        None, "distress", "Identify the content as a written message; no audio exists.",
        chat=[("user", "Estoy encerrado en el baño y no puedo salir. Por favor llama a mi madre.", -10)])
    assert len(rows) == 18 and len({r["scenario"] for r in rows}) == 18
    return rows
