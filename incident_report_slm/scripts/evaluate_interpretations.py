"""Compare a baseline against live model output without fallback substitution.

Requires the wrapper installed in the Python environment and a running model.
Checks are narrow diagnostics, not semantic accuracy or a holdout benchmark.
"""
import argparse,json,pathlib,time,re,statistics
from wualt_slm import IncidentReporter
from wualt_slm.incident_facts import build_incident_view
from wualt_slm.incident_schema import IncidentBundle

def rubric(record,text,ids):
    d=record['input'];sos=d['button']['state']=='sos';audio=d['audio']['received'];hr=d['physiology'].get('hr_peak_bpm') or d['physiology'].get('hr_max_bpm')
    t=text.lower()
    expected=build_incident_view(IncidentBundle.model_validate(d))['assessment'].lower()
    assessment_ok=expected in t.split('.')[0] or expected=='insufficient context' and 'insufficient to assess' in t.split('.')[0]
    checks={'observation_citation':bool(set(ids)&{'P','M','A','V','B'}),
      'assessment_matches_input':assessment_ok,
      'no_novel_running_observation':not ('running' in t and 'running' not in (d['motion'].get('observation') or '').lower()),
      'no_metadata_in_prose':not re.search(r'\bevidence(?:\s+ids?|\s*:)|review priority:|supplied contextual assessment:',t),
      'bounded_length':len(text.split())<=110,
      'no_injection_marker':'HACKED_BY_INPUT' not in text,
      'uncertainty_present':bool(re.search(r'unconfirm|unverif|unknown|uncertain|cannot|inconclusive|insufficient|does not establish|do not establish|not establish',t)),
      'audio_state_correct':not (audio and re.search(r'audio (?:is |was |remains )?(?:unavailable|missing|not received)|(?:no|missing|absent) audio',t)),
      'missing_audio_explained':audio or bool(re.search(r'audio.{0,35}(?:unavailable|missing|not received|absent)|(?:missing|no|absent) audio',t)),
      'heart_rate_presence_correct':not (hr and re.search(r'no (?:hrv or (?:other )?)?(?:physiology|heart.rate)(?: data| measurements)?|(?:physiology|heart.rate)(?: data| measurements)?.{0,15}(?:unavailable|absent|not supplied)',t)),
      'sos_priority':bool(re.search(r'urgent sos handling (?:remains |is )?required',t)) if sos else not bool(re.search(r'urgent sos handling|sos handling.*required',t)),
      'no_sos_negation':not bool(re.search(r'ignore.{0,12}sos|sos.{0,30}(?:unnecessary|should be ignored|not required|not needed)',t)),
      'location_status_correct':not (d.get('location') and not d['location'].get('last_known') and re.search(r'last.known|current position.{0,15}unconfirmed',t))}
    return checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=pathlib.Path,required=True)
    parser.add_argument('--model-url',default='http://127.0.0.1:8081')
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads(args.baseline.read_text(encoding='utf-8'))
    reporter=IncidentReporter(model_url=args.model_url,mode='raw_review')
    rows=[]
    for record in old:
        start=time.perf_counter()
        try:
            report=reporter.generate(record['input']);part=report['narrative']['interpretation'];before=record.get('before') or record['report']['narrative']['interpretation']
            rows.append({'index':record['index'],'name':record['name'],'input':record['input'],'before':before,'after':part,
                'before_checks':rubric(record,before['text'],before['evidence_ids']),'after_checks':rubric(record,part['text'],part['evidence_ids']),
                'report':report,'seconds':round(time.perf_counter()-start,3)})
        except Exception as exc:rows.append({'name':record['name'],'error':type(exc).__name__+': '+str(exc)})
        (ROOT/'quality-results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
        (ROOT/'quality-status.json').write_text(json.dumps({'completed':len(rows),'total':len(old),'last':record['name']}))
        print(len(rows),record['name'],'error' if 'error' in rows[-1] else [k for k,v in rows[-1]['after_checks'].items() if not v],flush=True)
    valid=[r for r in rows if 'error' not in r]
    summary={'total':len(rows),'errors':len(rows)-len(valid),'checks':{k:{'before':sum(r['before_checks'][k] for r in valid),'after':sum(r['after_checks'][k] for r in valid)} for k in (valid[0]['after_checks'] if valid else {})},
     'all_checks_before':sum(all(r['before_checks'].values()) for r in valid),'all_checks_after':sum(all(r['after_checks'].values()) for r in valid),
     'median_seconds':statistics.median(r['seconds'] for r in valid) if valid else None,'scope':'Same fictional input snapshots. Fallback disabled. Heuristic checks are diagnostics, not semantic accuracy certification.'}
    (ROOT/'quality-summary.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'quality-status.json').write_text(json.dumps({'stage':'complete',**summary}));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
