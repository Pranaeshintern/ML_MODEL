"""Run real inference and save each input, raw report, PDF and comparison HTML.

--validate-inputs-only prepares scenarios without claiming a model evaluation.
"""
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from context_scenarios import scenarios
from wualt_slm import IncidentBundle, IncidentReporter
from wualt_slm.backend import LlamaBackend


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def legacy_input(data):
    old=data.model_dump(mode='json',exclude={'activity','chat_context'})
    old.update(schema_version='incident-v1',request_id=old['request_id']+'-legacy',
               motion={'observation':data.activity.label if data.activity else None},
               audio={'received':bool(data.audio.transcript),'reliability':data.audio.transcript_reliability,
                      'transcript':[{'at':t.at.isoformat(),'text':t.text} for t in data.audio.transcript]})
    return old


def html_report(rows,summary):
    cards=[]
    def prose(report):
        return ''.join('<h4>'+escape(k.title())+'</h4><p>'+escape(v['text'])+'</p>'
                       for k,v in report['narrative'].items())
    for row in rows:
        columns=[]
        if row.get('before'):columns.append('<section><h3>Legacy model output</h3>'+prose(row['before'])+'</section>')
        if row.get('after'):columns.append('<section><h3>Contextual model output</h3>'+prose(row['after'])+'</section>')
        status='LIVE MODEL OUTPUT' if row.get('after') else 'INPUT VALIDATED · MODEL NOT RUN' if summary['inference']=='not_run' else 'GENERATION FAILED'
        card='<article><div class="status">'+status+'</div><h2>'+escape(row['name'])+'</h2><p class="focus">Review: '+escape(row['review_focus'])+'</p>'
        if row.get('error'):card+='<p class="error">'+escape(row['error'])+'</p>'
        if row.get('before_error'):card+='<p class="error">Legacy comparison failed: '+escape(row['before_error'])+'</p>'
        card+='<div class="columns">'+''.join(columns)+'</div>'
        if row.get('after'):
            card+='<p>Flags: '+escape('; '.join(row['after']['validation_issues']) or 'None detected; semantic review still required.')+'</p>'
            card+='<a href="reports/'+escape(row['name'])+'.pdf">Open contextual PDF</a>'
        card+='<details><summary>Input JSON</summary><pre>'+escape(json.dumps(row['input'],ensure_ascii=False,indent=2))+'</pre></details></article>'
        cards.append(card)
    subtitle=('Inputs are prepared for a future live run. No model reports or language-quality results have been generated.'
              if summary['inference']=='not_run' else 'Raw model text is retained. Diagnostic checks do not certify semantic accuracy; review each scenario against its input.')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SLM contextual report review</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#172638;background:#f2f5f7;margin:0}main{max-width:1200px;margin:auto;padding:40px 24px}h1{font-size:38px;line-height:1.15}h2{margin:8px 0}h3{font-size:17px;color:#375167}h4{margin-bottom:4px}article{background:white;border:1px solid #dae2e8;border-radius:12px;padding:25px;margin:22px 0}.status{font-size:11px;letter-spacing:1px;font-weight:700;color:#536b79}.focus{color:#4b5965}.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:28px}.columns p{white-space:pre-wrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:15px;font-size:12px}summary{cursor:pointer;padding-top:15px}.error{color:#9d2e26}a{color:#225b9b}.scope{padding:18px;background:#e6edf4;border-radius:8px}</style><main><h1>Contextual report review</h1><p class="scope">'''+escape(subtitle)+'</p><p>'+escape(json.dumps(summary,ensure_ascii=False))+'</p>'+''.join(cards)+'</main></html>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-url',default='http://127.0.0.1:18081')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare-legacy',action='store_true')
    parser.add_argument('--validate-inputs-only',action='store_true')
    args=parser.parse_args()
    backend=LlamaBackend(args.model_url)
    if not args.validate_inputs_only and not backend.ready():
        parser.error('The model is unavailable. No synthetic responses or fallback reports will be substituted.')
    root=args.output.resolve();(root/'inputs').mkdir(parents=True,exist_ok=True)
    (root/'reports').mkdir(exist_ok=True)
    reporter=IncidentReporter(backend=backend,mode='raw_review')
    rows=[];errors=0;run_id=uuid4().hex[:10]
    for case in scenarios():
        case['input']['request_id']+='-'+run_id
        data=IncidentBundle.model_validate(case['input'])
        row=dict(case);write_json(root/'inputs'/(case['name']+'.json'),case['input'])
        if not args.validate_inputs_only:
            start=time.monotonic()
            try:
                if args.compare_legacy:
                    try:
                        row['before']=reporter.generate(legacy_input(data))
                    except Exception as exc:
                        row['before_error']=type(exc).__name__+': '+str(exc)
                row['after']=reporter.generate(data)
                assert row['after']['fallback_enabled'] is False
                write_json(root/'reports'/(case['name']+'.json'),row['after'])
                (root/'reports'/(case['name']+'.pdf')).write_bytes(reporter.pdf(row['after']))
            except Exception as exc:
                row['error']=type(exc).__name__+': '+str(exc);errors+=1
            row['seconds']=round(time.monotonic()-start,3)
        rows.append(row)
        write_json(root/'results.json',rows)
        print(case['name'], 'input_validated' if args.validate_inputs_only else 'failed' if row.get('error') else 'generated',flush=True)
    summary={'scenarios':len(rows),'errors':errors,'inference':'not_run' if args.validate_inputs_only else 'live',
             'reports_generated':sum('after' in r for r in rows),'legacy_comparison_requested':args.compare_legacy,
             'legacy_errors':sum('before_error' in r for r in rows),
             'fallback_enabled':False,'model_identity_verified':False,
             'created_at':datetime.now(timezone.utc).isoformat(),
             'scope':'Fictional cases. Compare meaning manually; heuristic flags are diagnostics only.'}
    write_json(root/'summary.json',summary)
    with (root/'manual_review.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['scenario','review_focus','evidence_accuracy_0_to_2','context_used_0_to_2',
                         'relationships_explained_0_to_2','uncertainty_handled_0_to_2','question_useful_0_to_2','reviewer_notes'])
        for row in rows:
            writer.writerow([row['name'],row['review_focus'],'','','','','',''])
    (root/'review.html').write_text(html_report(rows,summary),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return 1 if errors or summary['legacy_errors'] else 0


if __name__=='__main__':
    raise SystemExit(main())
