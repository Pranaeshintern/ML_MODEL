"""Build the reference report's fields and evidence without model calculations."""
from zoneinfo import ZoneInfo

COLOURS={'RED':('#b62313','HIGH ABNORMALITY'),'YELLOW':('#f2c94c','MODERATE ABNORMALITY'),
         'GREEN':('#21673b','NO SIGNIFICANT ABNORMALITY DETECTED'),'GREY':('#586374','UNABLE TO ASSESS')}
ASSESSMENTS={'possible_distress':('RED','Possible distress'), 'exercise_consistent':('GREEN','Exercise-consistent'),
             'moderate_concern':('YELLOW','Moderate concern'), 'insufficient_context':('GREY','Insufficient context'),
             'inconclusive':('GREY','Inconclusive')}


def duration(seconds):
    seconds=int(seconds)
    minutes,sec=divmod(seconds,60)
    return (f'{minutes} min {sec} sec' if sec else f'{minutes} '+('minute' if minutes==1 else 'minutes')) if minutes else f'{sec} '+('second' if sec==1 else 'seconds')


def build_incident_view(data):
    zone=ZoneInfo(data.timezone)
    clock=lambda t:t.astimezone(zone).strftime('%H:%M:%S %Z')
    stamp=lambda t:t.astimezone(zone).strftime('%Y-%m-%d %H:%M:%S %Z (%z)')
    w=data.observation_window
    notes=[]
    colour,label=ASSESSMENTS[data.assessment.context]
    if not data.audio.received or data.audio.reliability in ('unavailable','unreliable'):
        colour='GREY';label='Insufficient context' if not data.audio.received else 'Inconclusive'
        notes.append('Audio is unavailable or unreliable; the interpretation remains unassessed.')
    if colour=='GREEN' and (data.button.state!='none_recorded' or not data.motion.observation or
                           not (data.physiology.hr_peak_bpm or data.physiology.hr_max_bpm) or data.audio.reliability!='usable'):
        colour='GREY';label='Insufficient context'
        notes.append('The supplied exercise assessment lacks sufficient context for the green label.')
    priority='Urgent SOS handling' if data.button.state=='sos' else 'Standard review' if colour=='GREEN' else 'Verification needed'
    button=f'SOS at {clock(data.button.at)}' if data.button.state=='sos' else 'No press recorded' if data.button.state=='none_recorded' else 'Button data unavailable'
    button_detail=button+'. '+(f'Press duration: {data.button.duration_seconds:g} seconds.' if data.button.duration_seconds else 'Press duration not supplied.') if data.button.state=='sos' else button+'.'
    p=data.physiology
    physiological=[]
    if p.hr_peak_bpm is not None:physiological.append(f'HR peak: {p.hr_peak_bpm} bpm.')
    if p.hr_min_bpm is not None:physiological.append(f'HR: {p.hr_min_bpm}-{p.hr_max_bpm} bpm.')
    if p.reference_median_bpm is not None:physiological.append(f'{"Resting" if p.reference_type=="resting" else "Pre-event"} median: {p.reference_median_bpm} bpm.')
    physiology=' '.join(physiological) or 'Heart-rate measurements not supplied.'
    peak=p.hr_peak_bpm or p.hr_max_bpm
    phys_interpretation='Measurements alone do not establish cause or current safety.'
    if p.hr_peak_bpm and p.reference_median_bpm and p.hr_peak_bpm>p.reference_median_bpm:
        phys_interpretation='Above the supplied reference; cause not established.'
    if data.button.state=='sos':phys_interpretation+=' Does not cancel an SOS.'
    motion=data.motion.observation or 'Motion data not supplied.'
    sound=(data.audio.sounds or 'Sound findings not supplied.') if data.audio.received else 'Audio not received; recording unavailable.'
    tone=(data.audio.tone or 'Voice findings not supplied.') if data.audio.received else 'Unavailable; no audio input.'
    sensor_rows=[
        {'signal':'Physiology','observation':physiology,'interpretation':phys_interpretation},
        {'signal':'Motion','observation':motion,'interpretation':'The supplied pattern is exercise-consistent; not proof of safety.' if colour=='GREEN' else 'Movement alone does not establish cause, safety or intent.'},
        {'signal':'Audio / sounds','observation':sound,'interpretation':'Missing audio is not silence; sounds cannot be assessed.' if not data.audio.received else 'Sound detections are contextual; source '+('reported as verified.' if data.audio.source_verified else 'not verified.')},
        {'signal':'Tone of voice','observation':tone,'interpretation':'No supported voice or emotional interpretation.' if not data.audio.received else 'Vocal cues only; emotional state is unconfirmed.'},
        {'signal':'Button press','observation':button_detail,'interpretation':'Recorded device SOS; intent remains unconfirmed.' if data.button.state=='sos' else 'No recorded SOS is not evidence of safety.' if data.button.state=='none_recorded' else 'Button status cannot be assessed.'},
    ]
    gps={'coordinates':'Not supplied','fix':'Location fix not supplied','url':None,'label':'Location unavailable','last_known':False}
    if data.location:
        loc=data.location;age=(data.incident_start-loc.fixed_at).total_seconds()
        relation=f'{duration(age)} before incident start' if age>0 else f'{duration(-age)} after incident start; not a start-time fix' if age<0 else 'Fix recorded at incident start'
        gps={'coordinates':f'Latitude: {abs(loc.latitude):.5f} {"N" if loc.latitude>=0 else "S"} | Longitude: {abs(loc.longitude):.5f} {"E" if loc.longitude>=0 else "W"}',
             'fix':f'{clock(loc.fixed_at)} | Accuracy: '+(f'+/-{loc.accuracy_m:g} m' if loc.accuracy_m is not None else 'not supplied')+f' | {relation}',
             'url':f'https://www.google.com/maps?q={loc.latitude:.5f},{loc.longitude:.5f}',
             'label':'Open last-known location in Google Maps | Current position unconfirmed' if loc.last_known else 'Open in Google Maps',
             'last_known':loc.last_known}
    if data.audio.received:
        speaker_count=str(data.audio.speaker_count) if data.audio.speaker_count is not None else 'Unavailable'
        if data.audio.speaker_count is not None and data.audio.reliability!='usable':speaker_count+=' (provisional)'
        genders='; '.join(f'{s.label}: {s.supplied_gender_label}' for s in data.audio.speakers)+' (unverified)' if data.audio.speakers else 'Not supplied'
        transcript=' '.join(f'{clock(t.at)}, {t.speaker or "speaker unassigned"}: "{t.text}"' for t in data.audio.transcript) or 'Transcript not supplied.'
        keywords='; '.join(data.audio.keywords) or 'Not supplied'
        speaker_note='Speaker count and gender labels are supplied estimates; '+('the wearer-to-speaker link is reported as verified by the server.' if data.audio.wearer_link_verified else 'the wearer-to-speaker link is unverified.')
    else:
        speaker_count='Unavailable';genders='Unavailable - audio not received';transcript='Unavailable - audio not received.';keywords='Unavailable - no transcript.'
        speaker_note='Missing speaker data does not mean zero speakers; no gender label is assigned.'
    limitations=list(data.limitations)
    if not data.audio.received:limitations.append('Audio and speaker data unavailable')
    elif data.audio.reliability!='usable':limitations.append('Audio/transcript reliability is limited')
    if not data.audio.wearer_link_verified and data.audio.received:limitations.append('Wearer-to-speaker link unverified')
    if data.location and data.location.last_known:limitations.append('Last-known location; current position unconfirmed')
    if not data.location:limitations.append('Location not supplied')
    if not physiological:limitations.append('Heart-rate data not supplied')
    limitations=list(dict.fromkeys(limitations))
    evidence=[]
    def add(key,text):evidence.append({'id':key,'text':text})
    for i,event in enumerate(sorted(data.events,key=lambda e:e.at)):
        add(f'E{i+1:02}',f'{clock(event.at)}: {event.observation}')
    for key,row in zip(('P','M','A','V','B'),sensor_rows):add(key,row['observation'])
    add('G',gps['fix']+(' Last-known location; current position unconfirmed.' if gps['last_known'] else ''))
    if data.audio.transcript:add('T',transcript+' Transcript reliability: '+data.audio.reliability+'. Speaker identity is not established by a supplied label.')
    add('C',f'Supplied contextual assessment: {label}. Review priority: {priority}. Colour: {colour}; illustrative, not a validated score.')
    add('L','; '.join(limitations) or 'No additional limitations supplied.')
    event_summary=' '.join(e['text'] for e in evidence if e['id'].startswith('E'))
    if not event_summary:event_summary=f'{button_detail} {motion} {physiology} {sound}'
    if colour=='RED':interpretation='Possible distress is indicated by the supplied contextual assessment. The recorded observations provide context but do not establish an attack, injury or cause. Speaker identity remains unconfirmed.'
    elif colour=='GREEN':interpretation='The supplied context is exercise-consistent. Recorded motion, audio and heart-rate observations provide context, but do not confirm the wearer is safe. An isolated sound does not establish danger.'
    elif colour=='YELLOW':interpretation='The supplied context indicates moderate concern requiring verification. The observations do not establish the cause, an injury or the wearer\'s safety.'
    else:interpretation='The available evidence is insufficient to assess the situation reliably. Missing or uncertain observations do not establish a harmless explanation or the wearer\'s safety.'
    if data.button.state=='sos':interpretation+=' An SOS is recorded; uncertainty does not cancel urgent SOS handling.'
    if not data.audio.received:interpretation+=' Missing audio is not evidence of silence.'
    if data.audio.reliability=='unreliable':interpretation+=' Transcript, tone and sound labels from one uncertain recording are not independent confirmations.'
    if gps['last_known']:interpretation+=' The last-known location does not establish the current position.'
    return {'report_number':data.report_number,'date_reported':data.reported_at.astimezone(zone).strftime('%d %B %Y'),
            'time_reported':clock(data.reported_at),'incident_start':stamp(data.incident_start),
            'window':f'{clock(w.start)} - {clock(w.end)} | {duration((w.end-w.start).total_seconds())} | {w.context or "Context not supplied"}',
            'gps':gps,'button':button,'priority':priority,'assessment':label,'assessment_provenance':data.assessment.provenance,
            'colour':colour,'colour_title':COLOURS[colour][1],'sensor_rows':sensor_rows,'speaker_count':speaker_count,
            'gender_labels':genders,'transcript':transcript,'keywords':keywords,'speaker_note':speaker_note,
            'limitations':limitations,'follow_up':data.follow_up or 'Use the configured review procedure; no procedure text was supplied.',
            'recorded_status':data.recorded_status or 'Not supplied. No action or response is inferred.',
            'evidence':evidence,'policy_notes':notes,'fallback_description':event_summary,'fallback_interpretation':interpretation}
