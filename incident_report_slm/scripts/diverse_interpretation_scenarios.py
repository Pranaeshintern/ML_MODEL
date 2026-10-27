"""Fictional, independently constructed situations; never generated interpretations."""
import copy
from datetime import datetime, timedelta


def diverse_cases():
    rows = []
    def add(key, title, group, speech, tone, state, focus, *, sounds=(), keywords=(), physiology=None, activity=None, chat=(), verified=False, reliability="usable", **changes):
        end = datetime.fromisoformat("2026-10-27T09:00:00+05:30") + timedelta(minutes=3 * len(rows))
        stamp = lambda seconds=0: (end + timedelta(seconds=seconds)).isoformat()
        payload = {"schema_version":"interpretation-v1", "request_id":"diverse-"+key,
                   "time":stamp(), "observation_window":{"start":stamp(-60),"end":stamp()},
                   "state":{"label":state,"confidence":0.82}, "transcript":speech,
                   "transcript_reliability":reliability if speech else "unavailable", "speaker_verified":verified,
                   "tone":{"label":tone,"confidence":0.85} if tone else None,
                   "keywords":{"stress":list(keywords) if state=="stress" else [],"distress":list(keywords) if state!="stress" else []},
                   "events":[{"label":label,"confidence":confidence,"at":stamp(-20)} for label,confidence in sounds],
                   "hr":None,"hrv":None,"location":None,"activity":activity,
                   "chat_context":[{"role":role,"text":text,"at":stamp(offset)} for role,text,offset in chat]}
        if physiology:
            bpm, variation, quality, comparable = physiology
            payload["hr"]={"bpm":bpm,"at":stamp(-15),"signal_quality":quality,"reference":{"bpm":76,"at":stamp(-600),"comparable_context":comparable}}
            payload["hrv"]={"metric":"rmssd","value_ms":variation,"window_start":stamp(-60),"window_end":stamp(),"signal_quality":quality,"artifact_corrected":True,"reference":{"metric":"rmssd","value_ms":40,"window_seconds":60,"at":stamp(-600),"comparable_context":comparable}}
        payload.update(copy.deepcopy(changes))
        rows.append({"scenario":key,"title":title,"category":group,
                     "description":("Speech supplied: "+speech) if speech else "No transcript was supplied; the other available signals must be assessed on their own.",
                     "review_focus":focus,"input":payload})
    raised=(118,23,"usable",True); unchanged=(76,40,"usable",True); poor=(126,20,"unreliable",True)
    g="Personal safety"
    add("following-concern","Concern about being followed",g,"Someone has followed me through three turns. I am scared. Please stay on the phone.","frightened","distress","Report the concern without confirming pursuit or an attack.",keywords=["scared","stay on the phone"],physiology=raised,activity="walking")
    add("following-denied","A person explicitly denies being followed",g,"Nobody is following me. I am just waiting for my taxi.","calm","normal","Preserve the denial despite the extracted keyword.",keywords=["following me"])
    add("unwanted-contact","Request to stop unwanted contact",g,"Please stop touching my arm. I said no. Let me leave.","firm and upset","distress","Explain the stated boundary and wish to leave; do not invent injury.",keywords=["said no","let me leave"],verified=True)
    add("argument-fear","Fear during an argument",g,"He is shouting at me and blocking the door. I am frightened.","shaky","distress","Attribute blocking and shouting to the report, not verified observation.",sounds=[("shouting",0.83)],physiology=raised)
    add("loud-disagreement","A loud disagreement without a reported threat",g,"We disagree about the bill. Nobody has threatened anyone.","irritated","stress","Do not turn disagreement into assault; preserve the denial.",sounds=[("raised voices",0.79)])
    add("phone-taken","A phone reportedly taken",g,"That person took my phone and ran away. Can you help me contact someone?","upset","distress","Describe the reported loss; do not invent pursuit or violence.",keywords=["help me"])
    add("door-pounding","Unidentified knocking at the door",g,"There is loud knocking. I do not know who it is, and I feel uneasy.","worried","stress","Unknown knocking does not establish a break-in.",sounds=[("knocking",0.94)])
    add("whispered-help","A quiet request for help",g,"I do not feel safe here. Please help me leave quietly.","whispering","normal","Quiet delivery and a normal label must not erase the help request.",keywords=["help me leave"],physiology=unchanged)
    g="Accidents and hazards"
    add("fall-self","A person reports falling",g,"I slipped on the steps. My ankle hurts and I need help getting up.","pained","distress","Report pain and the fall without diagnosing an injury.",sounds=[("impact",0.77)],verified=True)
    add("fall-other","A fall involving someone else",g,"My father fell in the kitchen. He says his knee hurts. I am calling for help.","concerned","distress","Keep the father's symptoms separate from the speaker and measurements.",physiology=raised)
    add("collision-report","A reported road collision",g,"Our car was hit from behind. We have stopped. I am shaken.","shaky","distress","Treat collision as a report; do not invent injuries.",sounds=[("impact",0.86)],physiology=raised)
    add("hard-braking","Sudden braking with an uncertain bang",g,"The bus braked suddenly. I heard a bang, but I cannot tell what happened.","startled","stress","Do not turn an uncertain sound into a confirmed crash.",sounds=[("collision",0.38)])
    add("smoke-stairwell","Smoke reported in a stairwell",g,"I can see smoke in the stairwell. I am frightened and looking for a clear exit.","urgent","distress","Report what the speaker says without inventing a confirmed fire or route.",sounds=[("alarm",0.9)],keywords=["smoke","frightened"],physiology=raised)
    add("alarm-drill","A known alarm drill",g,"This is the scheduled fire drill. We are walking outside as instructed.","calm","normal","An alarm in a stated drill does not establish an emergency.",sounds=[("alarm",0.96)],activity="walking")
    add("unusual-smell","An unexplained smell",g,"There is a strong smell like gas. I am worried, but I do not know its source.","concerned","stress","Do not confirm a gas leak or identify a source.")
    add("rising-water","Concern about rising water",g,"Water is entering this room and the door will not open. Please help.","urgent","distress","Explain the reported difficulty without inventing rescue or water depth.",sounds=[("rushing water",0.8)],keywords=["please help"])
    g="Reported physical symptoms"
    add("chest-discomfort","A report of chest discomfort",g,"My chest feels tight and I am worried. This started a few minutes ago.","worried","distress","Report symptoms without diagnosing their cause.",physiology=raised,verified=True)
    add("breathing-difficulty","Difficulty breathing is reported",g,"I am having trouble breathing and need someone to help me.","strained","distress","Explain the request and symptom without a diagnosis.",sounds=[("labored breathing",0.82)],physiology=raised,verified=True)
    add("dizzy-after-standing","Dizziness after standing up",g,"I stood up and felt dizzy. I am sitting down now.","uneasy","stress","Preserve the sequence; do not infer a medical mechanism.",physiology=unchanged)
    add("shaking-fear","Shaking and fear are reported",g,"My hands are shaking and I feel frightened, but I do not know why.","shaky","distress","An unexplained symptom does not establish panic disorder or a cause.",physiology=raised)
    add("headache-calm","A calm report of a headache",g,"I have a mild headache. I am otherwise feeling okay.","calm","normal","Keep the reported headache without inventing emotional distress.")
    add("pain-unchanged-heart","Pain despite unchanged heart measurements",g,"My wrist hurts a lot after I bumped it. I need help.","pained","distress","Unchanged heart readings must not invalidate reported pain.",physiology=unchanged,verified=True)
    add("symptoms-poor-signal","Symptoms with unreliable sensor readings",g,"I feel faint and nauseated. Can someone stay with me?","weak","distress","Use reported symptoms, not unreliable measurements, to explain concern.",physiology=poor)
    add("other-person-symptoms","A friend reports feeling unwell",g,"My friend says she feels dizzy. I am checking on her.","concerned","distress","Do not transfer the friend's symptom to the speaker.",physiology=raised)
    g="Everyday pressures"
    add("exam-nerves","Nervousness before an examination",g,"My exam starts soon and I am nervous about forgetting what I studied.","anxious","stress","Explain ordinary stated exam worry without inventing danger.",keywords=["nervous","exam"],physiology=raised)
    add("work-deadline","Pressure from a work deadline",g,"I have two hours to finish this work and I feel overwhelmed.","tense","stress","Tie stress to the stated workload, not a physical threat.",keywords=["overwhelmed"])
    add("interview-worry","Worry before a job interview",g,"I am waiting for my interview. I am nervous, but I want to try.","nervous","stress","Preserve both apprehension and intention.",physiology=raised)
    add("public-speaking","Anxiety before speaking to an audience",g,"I have to give a speech and my voice is shaking because I am nervous.","shaky","stress","Describe self-reported nervousness without a diagnosis.")
    add("missed-train","Frustration after missing a train",g,"I missed my train and now I will be late. This is frustrating.","frustrated","stress","Frustration is not evidence of immediate danger.",sounds=[("station announcement",0.8)])
    add("money-worry","Worry about paying bills",g,"I am worried about paying the rent this month. I cannot stop thinking about it.","worried","stress","Use the stated concern without predicting outcomes.")
    add("family-news","Distress after upsetting family news",g,"I just heard upsetting news about my family and I feel overwhelmed.","tearful","distress","Do not invent the content of the news.",sounds=[("crying",0.86)],physiology=raised)
    add("separated-child","A parent cannot find their child",g,"I cannot see my child in the crowd. I am very worried and asking for help.","frightened","distress","Report separation concern without inferring abduction.",sounds=[("crowd noise",0.9)],keywords=["worried","help"])
    g="Exercise and activity"
    add("stairs-exertion","Catching breath after climbing stairs",g,"I climbed six flights of stairs. I am catching my breath.","breathless","stress","Physical exertion can explain breathlessness without confirming emotional stress.",physiology=raised,activity="walking")
    add("planned-run","A planned run with a faster heartbeat",g,"I am halfway through my usual run and feeling good.","upbeat and breathless","normal","Do not turn exercise and faster heart rate into distress.",physiology=(142,22,"usable",False),activity="running",verified=True)
    add("cycling-hill","Effort while cycling uphill",g,"This hill is steep. I am working hard but enjoying the ride.","breathless","stress","Preserve enjoyment; do not assume emotional stress from effort.",physiology=(136,25,"usable",False),activity="cycling")
    add("heavy-bags","Effort from carrying heavy bags",g,"These grocery bags are heavy. I need to put them down for a moment.","strained","stress","Explain the stated physical effort without inventing injury.",activity="walking")
    add("workout-finished","Feeling pleased after a workout",g,"That workout was tough, but I feel great now.","cheerful","distress","Positive speech can conflict with the supplied distress label.",physiology=(130,26,"usable",False))
    add("sports-shouting","Shouting during a sports game",g,"Pass the ball! We can win this!","excited and loud","normal","Loud sports speech is not a request for emergency help.",sounds=[("cheering",0.92)],activity="running")
    add("dancing-celebration","Dancing at a celebration",g,"I love this song. We have been dancing for ages.","happy and breathless","stress","Do not mistake excitement and exertion for confirmed emotional stress.",sounds=[("music",0.95)],physiology=raised,activity="dancing")
    add("walking-chat-context","A walking label without audio",g,None,None,"normal","Activity alone does not establish emotional state or safety.",activity="walking",physiology=unchanged)
    g="Sounds and surroundings"
    add("thunder-startle","Being startled by thunder",g,"That thunder made me jump. I did not expect it.","startled","stress","Acknowledge the stated startle without inventing ongoing danger.",sounds=[("thunder",0.93)])
    add("fireworks-celebration","Fireworks at a celebration",g,"The fireworks have started. They look beautiful.","excited","normal","Preserve the stated celebration despite loud bangs.",sounds=[("explosion",0.61)])
    add("construction-bang","A loud sound attributed to construction",g,"The workers are drilling next door. That was a loud bang.","conversational","normal","Attribute construction to the speaker's account, not independent verification.",sounds=[("drilling",0.9)])
    add("tv-alarm","An alarm sound from a television",g,"That alarm is in the television show, not in this room.","calm","normal","Preserve the media context and negation.",sounds=[("alarm",0.92)])
    add("traffic-horns","Traffic noise without reported distress",g,"There is a lot of traffic. I am waiting at the crossing.","calm","normal","Noise does not establish a collision or danger.",sounds=[("car horn",0.91)],activity="standing")
    add("siren-no-speech","A siren with no speech",g,None,None,"distress","A siren alone does not identify an incident involving the subject.",sounds=[("siren",0.88)])
    add("glass-uncertain","An uncertain breaking sound",g,"I heard something break, but I cannot see what it was.","uncertain","stress","Do not invent a broken window or break-in.",sounds=[("glass breaking",0.55)])
    add("crying-no-words","Crying without an intelligible transcript",g,None,"tearful","distress","Crying may convey upset; its reason and speaker are unknown.",sounds=[("crying",0.89)])
    g="Meaning and attribution"
    add("happy-shout","A shout of happiness",g,"I got the job! I am so happy!","excited and loud","distress","Happy content conflicts with distress despite loudness.",sounds=[("shouting",0.86)],physiology=raised)
    add("tears-of-joy","Crying described as happiness",g,"I am crying because I am so happy to see you again.","tearful and warm","distress","Do not equate all crying with distress.",sounds=[("crying",0.88)])
    add("grief","A person describes grief",g,"I miss my brother so much since he died. I am feeling very sad today.","tearful","distress","Explain stated grief without diagnosis or unsupported immediate danger.")
    add("laughter-mislabel","Laughter during a pleasant conversation",g,"That story is hilarious. I cannot stop laughing.","amused","distress","Preserve the positive context and explain mismatch with the label.",sounds=[("laughter",0.96)])
    add("game-help-quote","The word help in a game",g,"My character is shouting help in the game. I am just playing.","relaxed","normal","A quoted game line is not the speaker's real-world plea.",keywords=["help"])
    add("past-incident","An upsetting event described as past",g,"I was scared during the power cut last week. Today I am okay.","reflective","distress","Do not convert past fear into current distress.",keywords=["scared"],physiology=raised)
    add("hypothetical-threat","A hypothetical safety question",g,"What should a character do if someone follows them in a story?","curious","distress","A hypothetical question is not a current report of being followed.",keywords=["follows"])
    add("friend-and-self-fear","Both a friend and the speaker report fear",g,"My friend is afraid of the shouting, and I am also scared.","frightened","distress","Do not suppress explicit first-person fear while handling third-party attribution.",sounds=[("shouting",0.84)])
    g="Chat context"
    add("chat-locked-room","A current chat message asks for help",g,None,None,"distress","Audio must be unavailable; distress may use the current user message.",chat=[("user","The door is jammed and I cannot get out. Please help.",-25)])
    add("chat-weather","Shivering attributed to cold in chat",g,"I am shivering.","strained","stress","Use current user context while avoiding a proven physiological cause.",chat=[("user","I am outside in cold rain and forgot my coat.",-30)],physiology=raised)
    add("chat-speech-conflict","Calm speech conflicts with a worried message",g,"Everything is fine.","calm","distress","Acknowledge conflicting channels; do not decide which is true.",chat=[("user","I feel unsafe but cannot speak freely.",-15)],verified=True)
    add("assistant-only-danger","Only the assistant asks about danger",g,None,None,"distress","An assistant question is not evidence of a user's distress.",chat=[("assistant","Are you hurt or in danger?",-20)])
    add("old-chat-issue","An old message describes an earlier problem",g,"I am having lunch now.","relaxed","distress","Do not treat old distress as a current incident.",chat=[("user","I was scared when the lift stopped.",-7200)])
    add("chat-recovery","A current message says the situation has changed",g,"I was frightened when the lights went out.","reflective","distress","Preserve recovery timing without certifying safety.",chat=[("user","The lights are back and I feel calm now.",-10)])
    add("chat-third-party","A message describes someone else's fear",g,None,None,"distress","Do not infer that the sender shares another person's fear.",chat=[("user","My sister is scared about her exam. I am helping her revise.",-20)])
    add("chat-spanish","A current Spanish message requests help",g,None,None,"distress","Use the current user's message; do not invent audio or measurements.",chat=[("user","Estoy atrapado en el ascensor y tengo miedo. Necesito ayuda.",-15)])
    g="Incomplete or conflicting data"
    add("empty-observations","No observations to explain the state",g,None,None,"distress","Return explicit insufficient-data statuses without fabricated paragraphs.")
    add("unknown-label-help","A help request with an unknown state",g,"I am frightened and need help leaving this place.","frightened","unknown","Explain the speech without treating unknown as a normal label.")
    add("panic-negated","A keyword appears inside a denial",g,"I am not panicking. I am concentrating on the instructions.","focused","normal","The keyword must not reverse the denial.",keywords=["panicking"])
    add("unverified-speaker-measurements","Speech and sensor readings may refer to different people",g,"I feel scared.","frightened","distress","Avoid assigning the unverified voice the measured heart changes.",physiology=raised,verified=False)
    add("not-comparable-measurements","Heart readings have an unsuitable reference",g,"I feel nervous about this meeting.","anxious","stress","Do not make a reference comparison when contexts are not comparable.",physiology=(112,21,"usable",False))
    add("calm-heart-explicit-fear","Explicit fear with unchanged measurements",g,"I am terrified that the door will not open.","frightened","distress","Unchanged measurements do not negate explicit fear.",physiology=unchanged,verified=True)
    add("noisy-transcript","An unreliable transcription",g,"...help...no...train...",None,"distress","Do not construct a story from unreliable fragments alone.",reliability="unreliable")
    add("uncertain-sound-label","A low-confidence scream label",g,None,None,"unknown","Explain uncertainty without treating detection as a confirmed person in distress.",sounds=[("scream",0.12)])
    g="Language coverage"
    add("hindi-help","Hindi request for help",g,"Mujhe darr lag raha hai. Darwaza nahi khul raha. Madad kijiye.","frightened","distress","Preserve fear and the stuck-door report without adding a cause.",keywords=["madad","darr"])
    add("tamil-help","Tamil request for help",g,"எனக்கு பயமாக இருக்கிறது. கதவு திறக்கவில்லை. உதவி செய்யுங்கள்.","frightened","distress","Explain the request in English without adding an incident.")
    add("spanish-fear","Spanish expression of fear",g,"Tengo miedo. Alguien está gritando y quiero salir.","frightened","distress","Preserve the reported shouting and wish to leave.",sounds=[("shouting",0.82)])
    add("arabic-help","Arabic request for help",g,"أنا خائف. الباب لا يفتح. أحتاج إلى مساعدة.","frightened","distress","Explain stated fear and need for help without claiming an attack.")
    add("tamil-english-mixed","Tamil and English mixed speech",g,"Enakku romba tension. Tomorrow interview irukku.","anxious","stress","Relate the stated tension to the interview without inventing danger.")
    add("french-normal","French everyday conversation",g,"Je vais retrouver mes amis pour déjeuner. Tout va bien.","calm","normal","Keep ordinary plans and the stated wellbeing without certifying safety.")
    add("hindi-negation","Hindi denial of fear",g,"Mujhe darr nahi lag raha. Main apne dost ka intezar kar raha hoon.","calm","normal","Preserve the negation rather than treating darr as positive evidence.",keywords=["darr"])
    add("language-unreliable","Unreliable multilingual fragments",g,"...ayuda...illa...maybe...",None,"unknown","Do not guess a coherent request from unreliable fragments.",reliability="unreliable")
    assert len(rows)==80
    return rows
