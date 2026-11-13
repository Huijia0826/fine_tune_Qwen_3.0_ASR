"""Trial ASR server (2026-11-04): our fine-tuned model behind an OpenAI-compatible endpoint + a test page.

    WF_SERVE_CONFIG=configs/serving/serve_D600.yaml .venv/bin/uvicorn whisperflow.serving.server:app \
        --host 127.0.0.1 --port 6006
    (from the Mac: ssh -N -L 7860:127.0.0.1:6006 -p <port> root@<autodl host>, then open http://localhost:7860)
    Qwen3-ASR (2026-11-12): WF_SERVE_CONFIG=configs/serving/serve_J3_600.yaml with .venv-qwen on the H100 cluster
    (scripts/serve_J3.sh; bind 0.0.0.0, tunnel to <compute node>:6006 through the login node).

Endpoints
  POST /v1/audio/transcriptions   multipart: file (wav / anything soundfile reads; other formats need ffmpeg),
                                  optional model, language (zh / en: force), prompt (Qwen only: context text, e.g.
                                  "Vocabulary: Vercel, PostHog"), response_format (json | text | verbose_json).
                                  Same shape as OpenAI's, so the VoiceCursor gateway can call it later.
  GET  /                          test page: record in the browser (16 kHz mono WAV made in the page), see the text,
                                  audio length and server time; history kept in the page.
  GET  /health                    model + decoding in use.
Decoding = the config's decoding block through the wf-transcribe transcriber of runtime.backend (make_transcriber):
Whisper (zh/en only, simplified output, loop guard, long-form fallback) or qwen3_asr (greedy, max_chunk_sec chunks,
simplified output), one request at a time (single GPU). One warm-up decode at startup (WF_SERVE_WARMUP=0 to skip) so
the first request's time is not CUDA start-up. Each request is logged to WF_SERVE_LOG (jsonl: time, audio length,
server ms, text) and, if WF_SERVE_SAVE_AUDIO=1, its audio is kept next to it for later error analysis.
Timing (2026-11-12): server_ms = model time only (transcribe call, as before). verbose_json / the log also carry
`timing`: total_ms (request handler start → response; the upload itself is not included), audio_decode_ms
(to 16 kHz mono), queue_ms (waiting for the previous request), model_ms, and with runtime.timing (Qwen) the model
split: features_ms, prefill_ms, decode_ms, post_ms, chunks, prompt_tokens, new_tokens. The page adds network =
round trip − total_ms (upload, SSH tunnel, response).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from whisperflow.data.audio import _ffmpeg_decode
from whisperflow.data.common import load_yaml
from whisperflow.eval.transcribe import make_transcriber

CFG_PATH = os.environ.get("WF_SERVE_CONFIG", "configs/serving/serve_D600.yaml")
LOG = Path(os.environ.get("WF_SERVE_LOG", "data/private/serve_logs/requests.jsonl"))
SAVE_AUDIO = os.environ.get("WF_SERVE_SAVE_AUDIO", "1") == "1"
SR = 16000
QWEN_LANG = {"zh": "Chinese", "en": "English"}

cfg = load_yaml(CFG_PATH)
BACKEND = cfg["runtime"].get("backend", "transformers")
tr = make_transcriber(cfg)
lock = threading.Lock()
app = FastAPI(title="Fine-tune Qwen ASR 3.0")
LOG.parent.mkdir(parents=True, exist_ok=True)


def to_16k_mono(data: bytes, name: str) -> np.ndarray:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr == SR:
            return x
    except Exception:  # not a format soundfile reads (webm, m4a, …): fall back to ffmpeg
        pass
    tmp = Path("/tmp") / f"wf_{uuid.uuid4().hex}_{Path(name).name}"
    tmp.write_bytes(data)
    try:
        return _ffmpeg_decode(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def run(audio: np.ndarray, language: str | None, prompt: str = "") -> tuple[str, dict, float, dict]:
    """Per-request overrides go into the transcriber's decoding dict and are restored afterwards (under the lock).
    Whisper: language → allowed_languages; Qwen: language → decoding.language, prompt → decoding.context."""
    dec = tr.decoding
    if BACKEND == "qwen3_asr":
        over = {"context": prompt or dec.get("context") or ""}
        if language in QWEN_LANG:
            over["language"] = QWEN_LANG[language]
    else:
        over = {"allowed_languages": [language]} if language in ("zh", "en") else {}  # force: LID picks only this
    t_wait = time.perf_counter()
    with lock:
        queue_ms = (time.perf_counter() - t_wait) * 1000
        saved = {k: dec.get(k) for k in over}
        dec.update(over)
        t0 = time.perf_counter()
        try:
            text = tr.transcribe([audio])[0]
        finally:
            dec.update(saved)
        ms = (time.perf_counter() - t0) * 1000
        timing = {"queue_ms": queue_ms, "model_ms": ms, **getattr(tr, "last_timing", {})}
    return text, (tr.last_info or [{}])[0], ms, timing


if os.environ.get("WF_SERVE_WARMUP", "1") == "1":
    run(np.zeros(SR, dtype=np.float32), None)


@app.get("/health")
def health() -> dict:
    return {"config": CFG_PATH, "backend": BACKEND, "model": cfg["model"], "decoding": {k: cfg["decoding"].get(k)
            for k in ("num_beams", "allowed_languages", "language", "max_chunk_sec", "output_script")},
            "dtype": cfg["runtime"].get("dtype")}


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form("whisperflow"),
                         language: str | None = Form(None), prompt: str = Form(""),
                         response_format: str = Form("json")):
    t_req = time.perf_counter()
    data = await file.read()
    audio = to_16k_mono(data, file.filename or "audio")
    decode_ms = (time.perf_counter() - t_req) * 1000
    text, info, ms, timing = await asyncio.to_thread(run, audio, language, prompt)
    dur = len(audio) / SR
    rid = uuid.uuid4().hex[:12]
    rec = {"id": rid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "audio_s": round(dur, 2), "server_ms": round(ms),
           "language": info.get("detected_language"), "forced_language": language or None, "prompt": prompt or None,
           "text": text, "backend": BACKEND, "config": CFG_PATH}
    if SAVE_AUDIO:
        sf.write(LOG.parent / f"{rid}.wav", audio, SR)
    timing = {"total_ms": (time.perf_counter() - t_req) * 1000, "audio_decode_ms": decode_ms, **timing}
    timing = {k: round(v) if k.endswith("_ms") else v for k, v in timing.items()}
    rec["timing"] = timing
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    headers = {"X-Server-Ms": str(round(ms)), "X-Audio-S": f"{dur:.2f}"}
    if response_format == "text":
        return PlainTextResponse(text, headers=headers)
    if response_format == "verbose_json":
        return JSONResponse({"text": text, "language": info.get("detected_language"), "duration": dur,
                             "server_ms": round(ms), "timing": timing, "id": rid}, headers=headers)
    return JSONResponse({"text": text}, headers=headers)


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Fine-tune Qwen ASR 3.0</title>
<style>body{font-family:-apple-system,"PingFang SC",sans-serif;max-width:860px;margin:30px auto;padding:0 20px;color:#1d1d1f}
button{font-size:18px;padding:14px 28px;border-radius:12px;border:0;background:#0071e3;color:#fff;cursor:pointer}
button.rec{background:#d70015}.muted{color:#666;font-size:13px}.item{background:#f5f5f7;border-radius:10px;padding:12px 16px;margin:10px 0}
.text{font-size:17px;line-height:1.6;white-space:pre-wrap}select,input{font-size:14px}</style></head><body>
<h2>Fine-tune Qwen ASR 3.0 <span class="muted" id="model"></span></h2>
<p><button id="btn">● 开始录音</button> &nbsp; 快捷键：空格 开始 / 结束 &nbsp;
语言 <select id="lang"><option value="">自动</option><option value="zh">中文</option><option value="en">English</option></select>
&nbsp; 或上传文件 <input type="file" id="file" accept="audio/*"></p>
<p id="vocabRow" style="display:none">词典（仅 Qwen，逗号分隔）<input id="vocab" size="60" placeholder="Vercel, PostHog, 王小明"></p>
<p class="muted">录音只发到你自己的服务器（经 SSH 隧道），不经过第三方。</p><div id="list"></div>
<script>
let ctx,src,proc,stream,buf=[],rec=false,t0;const btn=document.getElementById('btn');
let qwen=false;fetch('/health').then(r=>r.json()).then(h=>{qwen=h.backend==='qwen3_asr';const m=h.model.checkpoint_path||h.model.adapter_path||h.model.name_or_path;
document.getElementById('model').textContent=m+' · '+(qwen?'greedy · '+h.decoding.max_chunk_sec+' 秒分块':'beams '+h.decoding.num_beams)+' · '+h.dtype;
if(qwen)document.getElementById('vocabRow').style.display=''});
function wav(samples,sr){const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b);const w=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))};
w(0,'RIFF');v.setUint32(4,36+samples.length*2,true);w(8,'WAVE');w(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,samples.length*2,true);
for(let i=0;i<samples.length;i++){const s=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,s<0?s*0x8000:s*0x7fff,true)}return new Blob([b],{type:'audio/wav'})}
function resample(x,from,to){if(from===to)return x;const r=from/to,n=Math.floor(x.length/r),y=new Float32Array(n);
for(let i=0;i<n;i++){const a=Math.floor(i*r),b=Math.min(x.length,Math.floor((i+1)*r));let s=0;for(let j=a;j<b;j++)s+=x[j];y[i]=s/Math.max(1,b-a)}return y}
async function start(){stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false}});
ctx=new AudioContext();src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);buf=[];
proc.onaudioprocess=e=>buf.push(new Float32Array(e.inputBuffer.getChannelData(0)));src.connect(proc);proc.connect(ctx.destination);
rec=true;t0=performance.now();btn.textContent='■ 结束录音';btn.className='rec'}
async function stop(){rec=false;btn.textContent='● 开始录音';btn.className='';proc.disconnect();src.disconnect();stream.getTracks().forEach(t=>t.stop());
const n=buf.reduce((a,b)=>a+b.length,0),x=new Float32Array(n);let o=0;for(const b of buf){x.set(b,o);o+=b.length}
const sr=ctx.sampleRate;await ctx.close();send(wav(resample(x,sr,16000),16000),'rec.wav')}
function steps(t,rt){if(!t)return'';const p=[['网络/隧道',rt-t.total_ms],['音频解码',t.audio_decode_ms],['排队',t.queue_ms]];
if(t.prefill_ms!==undefined)p.push(['特征',t.features_ms],['prefill',t.prefill_ms],['decode',t.decode_ms],['后处理',t.post_ms]);else p.push(['模型',t.model_ms]);
let s=p.map(([k,v])=>k+' '+Math.round(v)+' ms').join(' · ');
if(t.new_tokens!==undefined)s+=' ｜ '+t.chunks+' 块 · 提示 '+t.prompt_tokens+' tok · 输出 '+t.new_tokens+' tok ('+Math.round(t.new_tokens/Math.max(t.decode_ms,1)*1000)+' tok/s)';
return'<div class="muted">'+s+'</div>'}
async function send(blob,name){const item=document.createElement('div');item.className='item';item.innerHTML='<div class="muted">识别中…</div>';
document.getElementById('list').prepend(item);const fd=new FormData();fd.append('file',blob,name);fd.append('response_format','verbose_json');
const l=document.getElementById('lang').value;if(l)fd.append('language',l);
const v=document.getElementById('vocab').value.split(/[,，、]/).map(s=>s.trim()).filter(Boolean);if(qwen&&v.length)fd.append('prompt','Vocabulary: '+v.join(', '));const t=performance.now();
try{const r=await fetch('/v1/audio/transcriptions',{method:'POST',body:fd});const j=await r.json();const rt=Math.round(performance.now()-t);
item.innerHTML='<div class="text"></div><div class="muted">音频 '+j.duration.toFixed(1)+' 秒 · 服务器 '+j.server_ms+' ms · 往返 '+rt+' ms · 语言 '+j.language+(fd.get('prompt')?' · 带词典':'')+' · '+new Date().toLocaleTimeString()+'</div>'+steps(j.timing,rt);
item.querySelector('.text').textContent=j.text;const a=document.createElement('audio');a.controls=true;a.src=URL.createObjectURL(blob);item.appendChild(a)}
catch(e){item.innerHTML='<div class="muted">出错：'+e+'</div>'}}
btn.onclick=()=>rec?stop():start();document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='SELECT'){e.preventDefault();rec?stop():start()}});
document.getElementById('file').onchange=e=>{const f=e.target.files[0];if(f)send(f,f.name)};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE
