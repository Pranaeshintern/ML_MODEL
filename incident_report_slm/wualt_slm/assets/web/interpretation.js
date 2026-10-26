"use strict";
const $=id=>document.getElementById(id);
let busy=false;
function headers(){const h={"Content-Type":"application/json"};if($("key").value)h["X-API-Key"]=$("key").value;return h;}
function fail(message){$("error").textContent=message;$("error").hidden=false;}
async function request(path,options={}){const response=await fetch(path,{...options,headers:headers()});const data=await response.json();if(!response.ok && !(path==="/v1/interpretations" && response.status===502 && data.sections)){const detail=data.detail;throw new Error(typeof detail==="string"?detail:JSON.stringify(detail));}return data;}
async function connect(){try{const health=await request("/health");$("model-state").textContent=health.model_ready?"Model ready":"Model unavailable";}catch(error){$("model-state").textContent="Connection failed";fail(error.message);}}
async function example(){try{$("payload").value=JSON.stringify(await request("/v1/interpretation-input-example"),null,2);$("error").hidden=true;}catch(error){fail(error.message);}}
$("connect").addEventListener("click",connect);
$("example").addEventListener("click",example);
const sectionNames=["audio","distress"];
const sectionMessages={
  no_audio_evidence:"No transcript, tone or sound events were supplied.",
  no_usable_audio_evidence:"No usable audio observations were supplied.",
  grounding_failed:"The generated paragraph failed validation and was not returned.",
  generation_incomplete:"The model did not finish this paragraph.",
  invalid_model_response:"The model returned an invalid response for this section.",
  model_unavailable:"The model service was unavailable for this section."
};
function validateResult(result){
  for(const name of sectionNames){
    const value=result[name+"_interpretation"],state=result.sections?.[name];
    if(!state || !["generated","not_available","failed"].includes(state.status))throw new Error("The service did not return section statuses. Use wrapper version 0.14 or later.");
    if(state.status==="generated" ? typeof value!=="string" || !value.trim() : value!==null)throw new Error("The service returned inconsistent interpretation text and status.");
    if(state.status!=="generated" && !sectionMessages[state.reason])throw new Error("The service did not explain why this section is unavailable.");
  }
}
$("generate").addEventListener("click",async()=>{
  if(busy)return;
  busy=true;$("generate").disabled=true;$("error").hidden=true;
  $("interpretations").hidden=true;
  for(const name of sectionNames){$(name+"-interpretation").textContent="";$(name+"-interpretation").hidden=true;$(name+"-status").textContent="";$(name+"-status").hidden=true;}
  $("copy").disabled=true;$("empty").hidden=false;
  $("status").textContent="Generating audio interpretation and distress explanation…";
  try{
    const input=JSON.parse($("payload").value);
    const result=await request("/v1/interpretations",{method:"POST",body:JSON.stringify(input)});
    validateResult(result);
    let generated=0;
    for(const name of sectionNames){
      const state=result.sections[name];
      if(state.status==="generated"){$(name+"-interpretation").textContent=result[name+"_interpretation"];$(name+"-interpretation").hidden=false;generated++;}
      else{$(name+"-status").textContent=(state.status==="not_available"?"Unavailable: ":"Generation failed: ")+sectionMessages[state.reason];$(name+"-status").hidden=false;}
    }
    $("interpretations").hidden=false;$("empty").hidden=true;$("copy").disabled=generated===0;
    $("status").textContent=generated===2?"Both interpretations are ready for review.":generated===1?"One interpretation is ready. The other section's status is shown separately.":"No interpretation was generated. Section statuses are shown below; no fallback was substituted.";
  }catch(error){fail(error.message);$("status").textContent="The interpretation request could not be completed.";}
  finally{busy=false;$("generate").disabled=false;}
});
$("copy").addEventListener("click",async()=>{
  const text=sectionNames.map(name=>(name==="audio"?"Audio interpretation":"Distress explanation")+"\n"+($(name+"-interpretation").hidden?$(name+"-status").textContent:$(name+"-interpretation").textContent)).join("\n\n");
  try{await navigator.clipboard.writeText(text);$("status").textContent="Interpretations and section statuses copied.";}
  catch{$("status").textContent="Select the interpretation text to copy it.";}
});
connect();example();
