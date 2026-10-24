"""Boxed incident form: supplied record fields and one model interpretation."""
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from .fonts import fonts, script_text
from .incident_facts import COLOURS


def boxed_contextual_story(report, width, size=9):
    d=report['input'];v=report['view'];n=report['narrative'];regular,bold=fonts()
    pdfmetrics.registerFontFamily(regular,normal=regular,bold=bold,italic=regular,boldItalic=bold)
    body=ParagraphStyle('BoxedBody',fontName=regular,fontSize=size,leading=size+2,spaceAfter=0,shaping=1)
    small=ParagraphStyle('BoxedSmall',parent=body,fontSize=8,leading=10)
    heading=ParagraphStyle('BoxedHeading',parent=body,fontName=bold,fontSize=9.5,leading=12)
    p=lambda value,sty=body:Paragraph(script_text(value),sty)
    def field(label,value):return Paragraph('<b>'+script_text(label)+':</b> '+script_text(value),body)
    def table(rows,widths=None,background=None):
        t=Table(rows,colWidths=widths or [width/len(rows[0])]*len(rows[0]),hAlign='LEFT')
        commands=[('GRID',(0,0),(-1,-1),.45,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),
                  ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                  ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
        if background:commands.append(('BACKGROUND',(0,0),(-1,-1),colors.HexColor(background)))
        t.setStyle(TableStyle(commands));return t
    story=[]
    def section(title,items,background='#c6dcf7',light=False):
        label=Paragraph(('<font color="white">' if light else '')+script_text(title)+('</font>' if light else ''),heading)
        bar=table([[label]],background=background)
        story.extend([Spacer(1,4),KeepTogether([bar,items[0]])]+items[1:])
    def lines(text):return [p(line) for line in text.splitlines()] or [p('Not supplied')]
    def banner(page_label):
        story.append(Paragraph('<b>WUALT INCIDENT REPORT</b> | '+script_text(page_label),heading))
        if d.get('synthetic'):story.append(p('FICTIONAL SCENARIO | MODEL OUTPUT FOR MANUAL REVIEW | FALLBACK DISABLED',small))
        elif report.get('report_mode') == 'raw_review':story.append(p('MODEL OUTPUT FOR MANUAL REVIEW | FALLBACK DISABLED',small))
        if report.get('preview_only'):story.append(p('LAYOUT PREVIEW - SIMULATED TEXT; NOT A LIVE MODEL RESULT',small))
        story.append(p('Scenario: '+report.get('review_scenario',d['incident_id']),small))
        story.append(Spacer(1,4))
    banner('Incident details and source observations')
    officer=d['officer'];person=d['wearer']
    story.append(table([
        [field('Report Number',v['report_number']),field('Time Reported',v['time_reported'])],
        [field('Date Reported',v['date_reported']),field('Reporting Officer',officer['name'])],
        [field('Officer ID',officer['officer_id']),field('Officer Unit',officer['unit'])]]))
    section('INCIDENT INFORMATION',[table([
        [field('Incident Type',d['incident_type'])],[field('Incident Start Timestamp',v['incident_start'])],
        [field('Observation Window / Context',v['window'])],
        [field('GPS Coordinates',v['gps']['coordinates'])],[field('GPS Fix',v['gps']['fix'])]])])
    section('WEARER / GUARDIAN / ALERT INFORMATION',[table([
        [field('Wearer ID',person['wearer_id']),field('Wearer Contact Number',person['contact'])],
        [field('Guardian',person['guardian_name']),field('Guardian Contact Number',person['guardian_contact'])]]),
        table([[field('Button Press',v['button'])],[field('Review Priority',v['priority']+' | Supplied assessment: '+v['assessment'])]])])
    section('INCIDENT DESCRIPTION',[table([[p(n['description']['text'])]])])
    rows=[[p('Signal',heading),p('Supplied observation',heading)]]
    for row in v['sensor_rows']:
        rows.append([p(row['signal']),p(row['observation'].replace('seconds.Current','seconds. Current'),small)])
    t=table(rows,[width*.20,width*.80]);t.repeatRows=1
    section('SENSOR FINDINGS',[t])
    section('AUDIO TRANSCRIPT / SOURCE INFORMATION',[table([
        [field('Transcription reliability',v['transcript_reliability'])],
        *[[paragraph] for paragraph in lines(v['transcript'])],
        [field('Keywords (derived hints)',', '.join(v.get('keywords',[])) or 'None supplied')],
        [p('Transcript speakers are not verified as the wearer. Keywords are extraction hints, not independent evidence.',small)]])])
    story.append(PageBreak())
    banner('SLM interpretation and supplied follow-up')
    story.append(table([[field('Report Number',v['report_number']),field('Review Priority',v['priority'])]]))
    section('CHAT MESSAGES',[table([[paragraph] for paragraph in lines(v['chat_context'])])])
    colour=v['colour']
    section('SLM INTERPRETATION | '+colour+' - '+v['colour_title'],[table([[p(n['interpretation']['text'])]])],COLOURS[colour][0],colour!='YELLOW')
    story.append(Spacer(1,3))
    story.append(Paragraph('<font color="#b62313"><b>RED: high</b></font> | <font color="#8a6800"><b>YELLOW: moderate</b></font> | <font color="#21673b"><b>GREEN: no significant abnormality</b></font> | <font color="#586374"><b>GREY: unable to assess</b></font>',small))
    story.append(p('Colour follows the supplied assessment and recorded SOS priority; it is not a validated model score. Green does not confirm safety.',small))
    section('CLARIFICATION REQUIRED (SUPPLIED)',[table([[p(n['clarification']['text'])]])])
    evidence_rows=[[field('Limits','; '.join(v['limitations']))],
                   [field('Follow-up',d.get('follow_up') or 'Not supplied')],
                   [field('Recorded Status',d.get('recorded_status') or 'Not supplied; no response action is inferred')]]
    if d.get('events'):
        evidence_rows.extend([[field('Recorded event',e['at']+': '+e['observation'])] for e in d['events']])
    if d.get('limitations'):evidence_rows.append([field('Additional source limits','; '.join(d['limitations']))])
    if report.get('validation_issues'):
        evidence_rows.append([field('Model review flags','; '.join(report['validation_issues']))])
    section('EVIDENCE / FOLLOW-UP',[table(evidence_rows)])
    story.append(Spacer(1,6))
    story.append(p('Officer Signature: __________________    Acknowledgement: '+(d.get('acknowledgement') or 'Not recorded')+'    Date: __________',small))
    story.append(Spacer(1,4))
    story.append(p('Model: '+report.get('model_id','Not recorded')+' | Prompt: '+report.get('prompt_version','Not recorded'),small))
    return story
