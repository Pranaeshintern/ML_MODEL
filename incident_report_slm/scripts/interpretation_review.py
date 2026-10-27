"""Readable standalone HTML for model results; never rewrite model prose."""
from collections import Counter
from pathlib import Path
import html
import json


def save_html(rows, target, metadata=None):
    metadata = metadata or {}
    esc = lambda value: html.escape(str(value), quote=True)
    counts = Counter(s["status"] for row in rows for s in row.get("output", {}).get("sections", {}).values())
    pending = sum("output" not in row and "error" not in row for row in rows)
    errors = sum("error" in row for row in rows)
    categories = list(dict.fromkeys(row.get("category", "Scenarios") for row in rows))
    cards=[]
    notices={"insufficient_context":"There is not enough reliable information to explain the supplied state.",
             "no_audio_evidence":"There is not enough audio data to provide an interpretation.",
             "no_usable_audio_evidence":"There is not enough reliable audio data to provide an interpretation."}
    for index,row in enumerate(rows,1):
        payload=row["input"];category=row.get("category","Scenarios");title=row.get("title",row["scenario"].replace("-"," ").capitalize())
        sections=[];output=row.get("output")
        for name,heading in (("audio","Audio interpretation"),("distress","Distress explanation")):
            state=(output or {}).get("sections",{}).get(name,{})
            if state.get("status")=="generated":
                body="<p class='interpretation'>"+esc(output[name+"_interpretation"])+"</p>"
            elif state.get("status")=="not_available":
                body="<p class='notice'>"+esc(notices[state["reason"]])+"</p>"
            elif state.get("status")=="failed" or "error" in row:
                body="<p class='notice failure'>The model did not produce an acceptable interpretation for this section. This is a generation failure, not evidence that data is missing.</p>"
            else:
                body="<p class='notice pending'>Not run yet. No model interpretation has been generated for this scenario.</p>"
            if state.get("status")=="failed" and state.get("diagnostics"):
                messages=[issue for attempt in state["diagnostics"] for issue in attempt["issues"]]
                body+="<details><summary>Why this section failed</summary><ul>"+"".join("<li>"+esc(issue)+"</li>" for issue in messages)+"</ul></details>"
            sections.append("<section><h3>"+heading+"</h3>"+body+"</section>")
        status="pending" if pending and not output and "error" not in row else "failed" if "error" in row or any(s["status"]=="failed" for s in (output or {}).get("sections",{}).values()) else "insufficient" if all(s["status"]=="not_available" for s in output["sections"].values()) else "generated"
        search=" ".join([title,category,row["scenario"],row.get("description", "")]).lower()
        notes=row.get("review_notes",[])
        note_html=("<aside class='review-note'><strong>Reviewer note</strong><ul>"+"".join("<li>"+esc(note)+"</li>" for note in notes)+"</ul></aside>") if notes else ""
        cards.append("<article id='"+esc(row["scenario"])+"' data-review='"+("yes" if notes else "no")+"' data-category='"+esc(category)+"' data-status='"+status+"' data-search='"+esc(search)+"'><div class='case-meta'><span>SCENARIO "+str(index).zfill(3)+"</span><span>"+esc(category)+"</span></div><h2>"+esc(title)+"</h2><p class='setup'><strong>Fictional input:</strong> "+esc(row.get("description","Review the supplied observations below."))+"</p><p class='state'>State supplied with the input: <strong>"+esc(payload["state"]["label"])+"</strong></p><div class='interpretations'>"+"".join(sections)+"</div>"+note_html+"<details><summary>View exact input JSON</summary><p class='small'>Original backend payload for this scenario. Scenario titles and descriptions are review labels and are not sent to the model.</p><pre>"+esc(json.dumps(payload,indent=2,ensure_ascii=False))+"</pre></details></article>")
    options="".join("<option>"+esc(category)+"</option>" for category in categories)
    both=sum(sum(s["status"]=="generated" for s in row.get("output",{}).get("sections",{}).values())==2 for row in rows)
    no_data=sum(bool(row.get("output")) and all(s["status"]=="not_available" for s in row["output"]["sections"].values()) for row in rows)
    title="Scenario catalogue — awaiting model run" if pending==len(rows) else "Audio and distress interpretation reports"
    intro="Prepared inputs for a new model run. The notices below are not generated interpretations." if pending else "Each scenario shows what was supplied and the two separate model interpretations. Missing-data notices and generation failures are shown explicitly."
    one=sum(sum(s["status"]=="generated" for s in row.get("output",{}).get("sections",{}).values())==1 for row in rows)
    stats=[(len(rows),"scenarios"),(both,"both interpretations"),(one,"one interpretation"),(no_data,"insufficient data"),(counts["failed"],"rejected sections")]
    noted=sum(bool(row.get("review_notes")) for row in rows)
    if noted:stats.append((noted,"reviewer notes"))
    if pending:stats.append((pending,"not run yet"))
    if errors:stats.append((errors,"request errors"))
    chips="".join("<div class='stat'><strong>"+str(value)+"</strong><span>"+label+"</span></div>" for value,label in stats)
    provenance=esc(metadata.get("provenance","Fictional evaluation inputs. Generated paragraphs are preserved exactly; no fallback interpretations are substituted."))
    model=esc(metadata.get("model","Model run pending" if pending else "See separate run metadata for model provenance"))
    document="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WUALT interpretation review</title><style>
:root{color-scheme:light;--ink:#193d32;--muted:#587466;--line:#d7e4dc;--green:#1c6448}*{box-sizing:border-box}body{margin:0;background:#f2f6f3;color:var(--ink);font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:42px 28px}header{padding-bottom:28px;border-bottom:2px solid var(--line)}.eyebrow,.case-meta{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}h1{font:650 36px/1.2 system-ui;letter-spacing:-.025em;margin:12px 0 18px;max-width:830px}header p{max-width:850px;color:var(--muted)}.stats{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.stat{border:1px solid var(--line);background:white;border-radius:9px;padding:12px 18px;min-width:140px}.stat strong{display:block;font-size:27px;line-height:1.2}.stat span{font-size:12px;color:var(--muted)}.filters{display:flex;gap:14px;flex-wrap:wrap;margin:25px 0 8px}.filters label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}input,select,button{font:inherit;padding:10px 13px;border:1px solid #afcabb;border-radius:6px;background:white;color:var(--ink)}input{min-width:260px}button{background:var(--green);color:white;border:0;cursor:pointer}.small,#visible-count{font-size:12px;color:var(--muted)}article{background:white;border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin:24px 0;box-shadow:0 3px 13px #20422d05}.case-meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}h2{font:650 24px/1.3 system-ui;letter-spacing:-.02em;margin:12px 0}.setup{color:#416154;font-size:14px;background:#f5f8f6;padding:13px 16px;border-radius:6px}.state{font-size:13px;color:var(--muted);padding-bottom:15px;border-bottom:1px solid var(--line)}h3{font:700 13px/1.4 system-ui;letter-spacing:.06em;text-transform:uppercase;color:var(--green);margin:23px 0 12px}.interpretation{font:18px/1.8 Georgia,serif;margin:0;white-space:pre-wrap}.notice{background:#fff5df;border-left:3px solid #c59832;padding:12px 16px;font-size:15px}.review-note{margin:22px 0;padding:14px 17px;border:1px solid #e5cf8c;border-radius:7px;background:#fff9e9;font:14px/1.6 system-ui}.review-note ul{margin:7px 0 0;padding-left:20px}.failure{background:#fff0ef;border-color:#b85a4f}.pending{background:#eef3f8;border-color:#7a91aa}details{margin-top:24px;border-top:1px solid var(--line);padding-top:16px}summary{cursor:pointer;font-size:14px;font-weight:650;color:var(--green)}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f5;padding:18px;border:1px solid var(--line);border-radius:7px;font:12px/1.6 Consolas,monospace}footer{border-top:1px solid var(--line);margin-top:35px;padding-top:20px;font-size:12px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:620px){main{padding:24px 14px}h1{font-size:28px}article{padding:22px 18px}.interpretation{font-size:17px}.stat{min-width:120px}}@media print{@page{margin:16mm}body{background:white}main{max-width:none;padding:0}article{break-inside:avoid;box-shadow:none}h1{font-size:26px}.interpretation{font-size:12pt;line-height:1.65}.filters,button,#visible-count{display:none}details pre{font-size:9pt}}
</style></head><body><main><header><div class="eyebrow">WUALT · FICTIONAL SCENARIO REVIEW</div><h1>"""+esc(title)+"</h1><p>"+intro+"</p><div class='stats'>"+chips+"</div><p class='small'>A generated interpretation is not a verified account of what happened. The supplied normal, stress or distress label is input to assess, not proof.</p><button type='button' onclick='window.print()'>Print / save as PDF</button></header><div class='filters'><label>Find a scenario<input id='search' type='search' placeholder='For example: stairs, chat, Tamil'></label><label>Situation<select id='category'><option value=''>All situations</option>"+options+"</select></label><label>Result<select id='status'><option value=''>All results</option><option value='generated'>Has an interpretation</option><option value='review'>Has a reviewer note</option><option value='insufficient'>Insufficient data</option><option value='failed'>Generation failure</option><option value='pending'>Not run yet</option></select></label></div><p id='visible-count'></p><div id='cards'>"+"".join(cards)+"</div><footer><p>"+model+"</p><p>"+provenance+"</p><p>Audio interpretations use only audio evidence. Distress explanations use the original validated supporting context. Passing automatic checks does not certify semantic accuracy.</p></footer></main><script>const search=document.getElementById('search'),category=document.getElementById('category'),status=document.getElementById('status'),cards=[...document.querySelectorAll('article')];function filter(){let visible=0;for(const card of cards){const show=(!search.value||card.dataset.search.includes(search.value.toLowerCase()))&&(!category.value||card.dataset.category===category.value)&&(!status.value||(status.value==='review'?card.dataset.review==='yes':card.dataset.status===status.value));card.hidden=!show;if(show)visible++;}document.getElementById('visible-count').textContent=visible+' of '+cards.length+' scenarios shown';}for(const control of [search,category,status])control.addEventListener('input',filter);filter();</script></body></html>"
    Path(target).write_text(document,encoding="utf-8")
