"""A4 renderer matching the supplied boxed, colour-coded sample layout."""
from io import BytesIO
from xml.sax.saxutils import escape
from . import settings
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepTogether
from .fonts import fonts,script_text
from .incident_facts import COLOURS

WIDTH=A4[0]-44


def report_story(report,size):
    data=report['input'];v=report['view'];n=report['narrative'];regular,bold=fonts()
    pdfmetrics.registerFontFamily(regular,normal=regular,bold=bold,italic=regular,boldItalic=bold)
    style=ParagraphStyle('Incident',fontName=regular,fontSize=size,leading=size+1.6,spaceAfter=0,splitLongWords=True,shaping=1)
    small=ParagraphStyle('Small',parent=style,fontSize=size-1,leading=size+.4)
    heading=ParagraphStyle('Heading',parent=style,fontName=bold)
    def text(value):return script_text(value)
    def p(value,sty=style):return Paragraph(value,sty)
    def field(label,value):return p(f'<b>{text(label)}:</b> {text(value)}')
    def table(rows,widths=None,background=None):
        t=Table(rows,colWidths=widths or [WIDTH],hAlign='LEFT')
        commands=[('GRID',(0,0),(-1,-1),.5,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),
                  ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                  ('TOPPADDING',(0,0),(-1,-1),2.5),('BOTTOMPADDING',(0,0),(-1,-1),2.5)]
        if background:commands.append(('BACKGROUND',(0,0),(-1,-1),colors.HexColor(background)))
        t.setStyle(TableStyle(commands));return t
    def bar(title,background='#c6dcf7',light=False):
        txt=f'<font color="white">{text(title)}</font>' if light else text(title)
        return table([[p(txt,heading)]],background=background)
    def section(title,content,background='#c6dcf7',light=False):
        story.extend([Spacer(1,2),KeepTogether([bar(title,background,light),Spacer(1,1),content[0]])]+content[1:])
    officer=data['officer'];person=data['wearer'];story=[]
    if report.get('report_mode')=='raw_review':
        story.append(p('<b>RAW MODEL OUTPUT - MANUAL REVIEW - FALLBACK DISABLED</b>',small))
        story.append(p('Model wording is retained even when validation flags it.'+(' This is a fictional test report.' if data['synthetic'] else ''),small))
        if report.get('review_scenario'):story.append(field('Test scenario',report['review_scenario']))
    story.append(table([
        [field('Report Number',v['report_number']),field('Time Reported',v['time_reported'])],
        [field('Date Reported',v['date_reported']),field('Reporting Officer',officer['name'])],
        [field('Officer ID',officer['officer_id']),field('Officer Unit',officer['unit'])],
    ],[WIDTH/2,WIDTH/2]))
    location=p('<b>Location:</b> '+(f'<link href="{text(v["gps"]["url"])}" color="#0758a4">{text(v["gps"]["label"])}</link>' if v['gps']['url'] else text(v['gps']['label'])))
    section('INCIDENT INFORMATION',[table([[field('Incident Type',data['incident_type'])],
        [field('Incident Start Timestamp',v['incident_start'])],[field('Observation Window / Context',v['window'])],
        [location],[field('GPS Coordinates',v['gps']['coordinates'])],[field('GPS Fix',v['gps']['fix'])]])])
    section('WEARER / GUARDIAN / ALERT INFORMATION',[
        table([[field('Wearer ID',person['wearer_id']),field('Wearer Contact Number',person['contact'])],
               [field('Guardian',person['guardian_name']),field('Guardian Contact Number',person['guardian_contact'])]],[WIDTH/2,WIDTH/2]),
        table([[field('Button Press',v['button'])],[field('Review Priority',v['priority']+' | Assessment: '+v['assessment'])]])])
    section('INCIDENT DESCRIPTION',[table([[p(text(n['description']['text']))]])])
    sensor=[[p('<b>Signal</b>'),p('<b>Observation</b>'),p('<b>Interpretation</b>')]]
    sensor.extend([[p(text(r['signal'])),p(text(r['observation'])),p(text(r['interpretation']))] for r in v['sensor_rows']])
    sensor_table=table(sensor,[WIDTH*.16,WIDTH*.42,WIDTH*.42]);sensor_table.repeatRows=1
    section('SENSOR FINDINGS',[sensor_table])
    section('AUDIO TRANSCRIPT / SPEAKER INFORMATION',[
        table([[p('<b>Speakers Detected:</b> '+text(v['speaker_count'])+' | <b>Gender Labels (supplied):</b> '+text(v['gender_labels']))],
               [field('Audio Transcript',v['transcript'])],[field('Keywords',v['keywords'])]]),
        Spacer(1,2),p(text(v['speaker_note']),small)])
    colour=v['colour'];section('SLM INTERPRETATION | '+colour+' - '+v['colour_title'],
        [table([[p(text(n['interpretation']['text']))]])],COLOURS[colour][0],colour!='YELLOW')
    story.extend([Spacer(1,3),p('<font color="#b62313"><b>RED: high</b></font> | <font color="#8a6800"><b>YELLOW: moderate</b></font> | <font color="#21673b"><b>GREEN: no significant abnormality</b></font> | <font color="#586374"><b>GREY: unable to assess</b></font>',small),
                  p('Green does not confirm safety. Unassessed data does not cancel an SOS.',small)])
    section('EVIDENCE / FOLLOW-UP',[table([[p('<b>Limits:</b> '+text('; '.join(v['limitations']) or 'Not supplied')+'<br/><b>Follow-up:</b> '+text(v['follow_up'])+'<br/><b>Recorded Status:</b> '+text(v['recorded_status']))]])])
    story.extend([Spacer(1,5),p('Officer Signature: __________________ &nbsp; Acknowledgement: '+text(data.get('acknowledgement') or 'Not recorded')+' &nbsp; Date: __________',small)])
    return story


def render_incident_pdf(reports):
    if isinstance(reports,dict):reports=[reports]
    stream=BytesIO();regular,_=fonts();story=[]
    for index,report in enumerate(reports):
        if index:story.append(PageBreak())
        for size in (8.8,8.5,8.2,8.0):
            parts=report_story(report,size)
            def height(item):
                if isinstance(item,KeepTogether):return sum(height(i) for i in item._content)
                return item.wrap(WIDTH,10000)[1]
            if sum(height(i) for i in parts)<=A4[1]-62:break
        story.extend(parts)
    sample=all(r['input']['synthetic'] for r in reports)
    def footer(canvas,doc):
        canvas.setFont(regular,6.5);canvas.setFillColor(colors.HexColor('#444444'))
        label='WUALT | '+('RAW MODEL - UNVALIDATED' if any(r.get('report_mode')=='raw_review' for r in reports) else 'FICTIONAL SAMPLE' if sample else 'DRAFT FOR REVIEW')+' | Colour labels are illustrative, not validated scores.'
        canvas.drawCentredString(A4[0]/2,18,label);canvas.drawRightString(A4[0]-22,8,str(doc.page))
    doc=SimpleDocTemplate(stream,pagesize=A4,leftMargin=16,rightMargin=16,topMargin=16,bottomMargin=34,title='WUALT Incident Report',author='WUALT')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return stream.getvalue()
