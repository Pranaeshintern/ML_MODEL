"use strict";
const $=id=>document.getElementById(id);
let busy=false;
function headers(){const h={"Content-Type":"application/json"};if($("key").value)h["X-API-Key"]=$("key").value;return h;}
function fail(message){$("error").textContent=message;$("error").hidden=false;}
async function request(path,options={}){const response=await fetch(path,{...options,headers:headers()});const data=await response.json();if(!response.ok){const detail=data.detail;throw new Error(typeof detail==="string"?detail:JSON.stringify(detail));}return data;}
async function connect(){try{const health=await request("/health");$("model-state").textContent=health.model_ready?"Model ready":"Model unavailable";}catch(error){$("model-state").textContent="Connection failed";fail(error.message);}}
async function example(){try{$("payload").value=JSON.stringify(await request("/v1/interpretation-input-example"),null,2);$("error").hidden=true;}catch(error){fail(error.message);}}
$("connect").addEventListener("click",connect);
$("example").addEventListener("click",example);
$("generate").addEventListener("click",async()=>{
  if(busy)return;
  busy=true;$("generate").disabled=true;$("error").hidden=true;
  $("interpretations").hidden=true;
  $("audio-interpretation").textContent="";$("distress-interpretation").textContent="";
  $("copy").disabled=true;$("empty").hidden=false;
  $("status").textContent="Generating audio interpretation and distress explanation…";
  try{
    const input=JSON.parse($("payload").value);
    const result=await request("/v1/interpretations",{method:"POST",body:JSON.stringify(input)});
    if(typeof result.audio_interpretation!=="string" || !result.audio_interpretation.trim() || typeof result.distress_interpretation!=="string" || !result.distress_interpretation.trim())throw new Error("The service did not return both interpretations. Check that the wrapper is version 0.13 or later.");
    $("audio-interpretation").textContent=result.audio_interpretation;
    $("distress-interpretation").textContent=result.distress_interpretation;
    $("interpretations").hidden=false;$("empty").hidden=true;$("copy").disabled=false;
    $("status").textContent="Both interpretations are ready for review.";
  }catch(error){fail(error.message);$("status").textContent="The interpretation pair was not generated.";}
  finally{busy=false;$("generate").disabled=false;}
});
$("copy").addEventListener("click",async()=>{
  const text="Audio interpretation\n"+$("audio-interpretation").textContent+"\n\nDistress explanation\n"+$("distress-interpretation").textContent;
  try{await navigator.clipboard.writeText(text);$("status").textContent="Both interpretations copied.";}
  catch{$("status").textContent="Select the interpretation text to copy it.";}
});
connect();example();
