"""Synthesise the TTS comparison set with stock voices (PAID API calls - run only with the owner's keys).

    uv run python scripts/tts_compare/synth.py --vendor minimax --dry-run       # count chars, no calls
    uv run python scripts/tts_compare/synth.py --vendor minimax --max-voices 6  # synthesise
    uv run python scripts/tts_compare/synth.py --vendor all                     # every vendor whose key is set

Vendors (keys come from environment variables only; see README.md):
    minimax           MiniMax Speech, POST {MINIMAX_API_BASE}/v1/t2a_v2 (hex audio in JSON)
    aliyun_cosyvoice  百炼 CosyVoice v3 via the official `dashscope` SDK, tts_v2 (pip install dashscope)
    aliyun_qwen       百炼 Qwen-TTS, POST .../services/aigc/multimodal-generation/generation (returns a WAV URL)
    volcengine        豆包语音合成 2.0, POST openspeech.bytedance.com/api/v3/tts/unidirectional (base64 chunks)
    azure             Azure Neural TTS, POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1

Output: data/private/tts_compare/<vendor>/<voice>/<id>.wav (16 kHz mono PCM16, resampled if the API returns
another rate) + data/private/tts_compare/<vendor>/manifest.jsonl (one row per sentence x voice, incl. errors).
Existing wavs are kept unless --force, so a re-run only fills gaps. Request shapes follow each vendor's docs as
of 2026-11-09; fields marked UNVERIFIED could not be tested without a key.

Voices: zh voices read every sentence (zh / mix / en - their English gives "Chinese speaker reading English");
en voices read only the en sentences.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import requests
import soundfile as sf

OUT_ROOT = Path("data/private/tts_compare")
SENTENCES = Path(__file__).with_name("sentences.txt")
TARGET_SR = 16000
TIMEOUT = 60


@dataclass(frozen=True)
class Voice:
    id: str
    label: str  # short human label for the listening page
    lang: str  # "zh" (reads all sentences) or "en" (reads en sentences only)
    style: str | None = None  # Azure express-as style, if the voice supports it


# Default stock voices per vendor. IDs were checked against the vendors' public voice lists on 2026-11-09
# (MiniMax CN list, Volcengine 2.0 list, 百炼 CosyVoice / Qwen-TTS lists, Azure language-support page).
# Order = priority; --max-voices N takes the first N zh and the first ceil(N/2) en voices.
VENDORS: dict[str, dict] = {
    "minimax": {
        "env": ["MINIMAX_API_KEY"],
        "voices": [
            Voice("male-qn-qingse", "青涩青年男", "zh"),
            Voice("female-shaonv", "少女", "zh"),
            Voice("Chinese (Mandarin)_Warm_Bestie", "温暖闺蜜女", "zh"),
            Voice("Chinese (Mandarin)_Straightforward_Boy", "直率男生", "zh"),
            Voice("Chinese (Mandarin)_Southern_Young_Man", "南方口音青年男", "zh"),
            Voice("Chinese (Mandarin)_Mature_Woman", "成熟女", "zh"),
            Voice("Chinese (Mandarin)_Humorous_Elder", "幽默老年男", "zh"),
            Voice("Chinese (Mandarin)_HK_Flight_Attendant", "港腔普通话女", "zh"),
            Voice("English_Trustworthy_Man", "en trustworthy man", "en"),
            Voice("English_Graceful_Lady", "en graceful lady", "en"),
            Voice("English_Aussie_Bloke", "en Aussie male", "en"),
            Voice("English_Diligent_Man", "en diligent man", "en"),
        ],
    },
    "aliyun_cosyvoice": {
        "env": ["DASHSCOPE_API_KEY"],
        "voices": [
            Voice("longxiaochun_v3", "龙小淳 知性女", "zh"),
            Voice("longyichen_v3", "龙逸尘 活力男", "zh"),
            Voice("longanyang", "龙安洋 阳光男孩", "zh"),
            Voice("longwan_v3", "龙婉 柔声女", "zh"),
            Voice("longsanshu_v3", "龙三叔 沉稳男", "zh"),
            Voice("longyingjing_v3", "龙应静 冷静女", "zh"),
            Voice("longlaotie_v3", "龙老铁 东北男", "zh"),
            Voice("longjielidou_v3", "龙杰力豆 10岁男孩", "zh"),
            Voice("loongandy_v3", "en Andy", "en"),
            Voice("loongbetty_v3", "en Betty", "en"),
            Voice("loongdavid_v3", "en David", "en"),
            Voice("loongemily_v3", "en Emily", "en"),
        ],
    },
    "aliyun_qwen": {
        "env": ["DASHSCOPE_API_KEY"],
        "voices": [
            Voice("Cherry", "Cherry 亲切女", "zh"),
            Voice("Ethan", "Ethan 北方口音男", "zh"),
            Voice("Serena", "Serena 温柔女", "zh"),
            Voice("Andre", "Andre 沉稳男", "zh"),
            Voice("Nofish", "Nofish 不卷舌男", "zh"),
            Voice("Eldric Sage", "Eldric 老者", "zh"),
            Voice("Dylan", "Dylan 北京话男", "zh"),
            Voice("Sunny", "Sunny 四川话女", "zh"),
            Voice("Jennifer", "Jennifer en female", "en"),
            Voice("Aiden", "Aiden en male", "en"),
        ],
    },
    "volcengine": {
        "env": ["VOLC_TTS_API_KEY|VOLC_TTS_APP_ID+VOLC_TTS_ACCESS_KEY"],
        "voices": [
            Voice("zh_female_vv_uranus_bigtts", "vv 女", "zh"),
            Voice("zh_male_m191_uranus_bigtts", "m191 男", "zh"),
            Voice("zh_female_xiaohe_uranus_bigtts", "小何 女", "zh"),
            Voice("zh_male_taocheng_uranus_bigtts", "taocheng 男", "zh"),
            Voice("zh_female_shuangkuaisisi_uranus_bigtts", "爽快思思 女", "zh"),
            Voice("zh_male_liufei_uranus_bigtts", "刘飞 男", "zh"),
            Voice("zh_female_popo_uranus_bigtts", "婆婆", "zh"),
            Voice("zh_male_kailangdidi_uranus_bigtts", "开朗弟弟", "zh"),
            Voice("en_male_tim_uranus_bigtts", "en Tim", "en"),
            Voice("en_female_dacey_uranus_bigtts", "en Dacey", "en"),
            Voice("en_female_stokie_uranus_bigtts", "en Stokie", "en"),
            Voice("en_male_michael_uranus_bigtts", "en Michael", "en"),
        ],
    },
    "azure": {
        "env": ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION|AZURE_SPEECH_ENDPOINT"],
        "voices": [
            Voice("zh-CN-XiaoxiaoNeural", "晓晓 chat", "zh", style="chat"),
            Voice("zh-CN-YunxiNeural", "云希 chat", "zh", style="chat"),
            Voice("zh-CN-XiaoxiaoMultilingualNeural", "晓晓 multilingual", "zh"),
            Voice("zh-CN-YunyiMultilingualNeural", "云逸 multilingual", "zh"),
            Voice("zh-CN-Xiaoxiao:DragonHDFlashLatestNeural", "晓晓 DragonHD Flash", "zh"),
            Voice("zh-CN-YunjianNeural", "云健 男", "zh"),
            Voice("zh-CN-sichuan-YunxiNeural", "云希 四川口音", "zh"),
            Voice("zh-CN-XiaoshuangNeural", "晓双 儿童", "zh", style="chat"),
            Voice("en-US-AvaMultilingualNeural", "en-US Ava", "en"),
            Voice("en-US-AndrewMultilingualNeural", "en-US Andrew", "en"),
            Voice("en-SG-WayneNeural", "en-SG Wayne", "en"),
            Voice("en-HK-YanNeural", "en-HK Yan", "en"),
        ],
    },
}


# ---------------------------------------------------------------- audio helpers
def to_16k_mono(audio: bytes, raw_sr: int | None = None) -> np.ndarray:
    """Decode a WAV/FLAC/MP3 container (or raw PCM16 LE if raw_sr is given) and return float32 16 kHz mono."""
    if raw_sr is not None:
        x, sr = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0, raw_sr
    else:
        x, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if not v:
        raise RuntimeError(f"environment variable {name} is not set")
    return v


def check_json(r: requests.Response) -> dict:
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


# ---------------------------------------------------------------- vendor calls (text, voice, lang) -> 16 kHz array
def call_minimax(text: str, v: Voice, lang: str) -> np.ndarray:
    base = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.cn")  # intl: https://api.minimax.io
    url = f"{base}/v1/t2a_v2"
    if os.environ.get("MINIMAX_GROUP_ID"):  # older accounts; current docs no longer require it (UNVERIFIED)
        url += f"?GroupId={os.environ['MINIMAX_GROUP_ID']}"
    body = {
        "model": os.environ.get("MINIMAX_MODEL", "speech-2.8-hd"),
        "text": text,
        "stream": False,
        "language_boost": "auto",
        "output_format": "hex",
        "voice_setting": {"voice_id": v.id, "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": TARGET_SR, "format": "wav", "channel": 1},
    }
    # pronunciation fixes, e.g. MINIMAX_PRON='["Supabase/(ˈsuːpəbeɪs)"]' (format "word/(replacement)")
    if os.environ.get("MINIMAX_PRON"):
        body["pronunciation_dict"] = {"tone": json.loads(os.environ["MINIMAX_PRON"])}
    r = requests.post(url, json=body, timeout=TIMEOUT,
                      headers={"Authorization": f"Bearer {env('MINIMAX_API_KEY')}"})
    j = check_json(r)
    if j.get("base_resp", {}).get("status_code", 0) != 0:
        raise RuntimeError(f"MiniMax error: {j.get('base_resp')}")
    return to_16k_mono(bytes.fromhex(j["data"]["audio"]))


def call_aliyun_cosyvoice(text: str, v: Voice, lang: str) -> np.ndarray:
    import dashscope  # official SDK: pip install dashscope
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

    dashscope.api_key = env("DASHSCOPE_API_KEY")
    if os.environ.get("DASHSCOPE_WS_URL"):  # intl: wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference
        dashscope.base_websocket_api_url = os.environ["DASHSCOPE_WS_URL"]
    model = os.environ.get("COSYVOICE_MODEL", "cosyvoice-v3-flash")
    synth = SpeechSynthesizer(model=model, voice=v.id, format=AudioFormat.WAV_16000HZ_MONO_16BIT)
    audio = synth.call(text)
    if not audio:
        raise RuntimeError(f"CosyVoice returned no audio (request id {synth.get_last_request_id()})")
    return to_16k_mono(audio)


def call_aliyun_qwen(text: str, v: Voice, lang: str) -> np.ndarray:
    base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")  # intl: dashscope-intl
    body = {
        "model": os.environ.get("QWEN_TTS_MODEL", "qwen3-tts-flash"),
        # language_type "Auto" for mixed text per docs ("无法保证完全精准"); UNVERIFIED which is best for zh-en
        "input": {"text": text, "voice": v.id,
                  "language_type": {"zh": "Chinese", "en": "English"}.get(lang, "Auto")},
    }
    r = requests.post(f"{base}/services/aigc/multimodal-generation/generation", json=body, timeout=TIMEOUT,
                      headers={"Authorization": f"Bearer {env('DASHSCOPE_API_KEY')}"})
    j = check_json(r)
    wav = requests.get(j["output"]["audio"]["url"], timeout=TIMEOUT)  # URL valid 24 h; likely 24 kHz WAV
    wav.raise_for_status()
    return to_16k_mono(wav.content)


def call_volcengine(text: str, v: Voice, lang: str) -> np.ndarray:
    headers = {
        "X-Api-Resource-Id": os.environ.get("VOLC_TTS_RESOURCE_ID", "seed-tts-2.0"),
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    if os.environ.get("VOLC_TTS_API_KEY"):  # new console
        headers["X-Api-Key"] = os.environ["VOLC_TTS_API_KEY"]
    else:  # old console: App ID + Access Token
        headers["X-Api-App-Id"] = env("VOLC_TTS_APP_ID")
        headers["X-Api-Access-Key"] = env("VOLC_TTS_ACCESS_KEY")
    additions = {"explicit_language": "en" if lang == "en" else "zh-cn"}  # zh-cn = 中文为主，支持中英混读
    if os.environ.get("VOLC_PRON"):  # e.g. VOLC_PRON='["Supabase/(ˈsuːpəbeɪs)"]' (UNVERIFIED IPA support)
        additions["pronunciation_dict"] = {"tone": json.loads(os.environ["VOLC_PRON"])}
    body = {
        "user": {"uid": "tts_compare"},
        "req_params": {
            "text": text,
            "speaker": v.id,
            "audio_params": {"format": "pcm", "sample_rate": TARGET_SR},
            "additions": json.dumps(additions, ensure_ascii=False),  # must be a JSON *string*
        },
    }
    r = requests.post("https://openspeech.bytedance.com/api/v3/tts/unidirectional", json=body,
                      headers=headers, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    # response = a stream of JSON objects {"code":0,"data":"<base64 pcm>"} ... {"code":20000000,...}
    dec, s, i, pcm = json.JSONDecoder(), r.text, 0, bytearray()
    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        obj, i = dec.raw_decode(s, i)
        code = obj.get("code", 0)
        if code not in (0, 20000000):
            raise RuntimeError(f"Volcengine error {code}: {obj.get('message')}")
        if obj.get("data"):
            pcm += base64.b64decode(obj["data"])
    if not pcm:
        raise RuntimeError("Volcengine returned no audio")
    return to_16k_mono(bytes(pcm), raw_sr=TARGET_SR)


def call_azure(text: str, v: Voice, lang: str) -> np.ndarray:
    endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT")  # e.g. https://chinanorth3.tts.speech.azure.cn/cognitiveservices/v1
    if not endpoint:
        endpoint = f"https://{env('AZURE_SPEECH_REGION')}.tts.speech.microsoft.com/cognitiveservices/v1"
    xml_lang = v.id.split("-")[0] + "-" + v.id.split("-")[1]
    inner = escape(text)
    if v.style:
        inner = f"<mstts:express-as style='{v.style}'>{inner}</mstts:express-as>"
    ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='{xml_lang}'>"
            f"<voice name='{v.id}'>{inner}</voice></speak>")
    r = requests.post(endpoint, data=ssml.encode("utf-8"), timeout=TIMEOUT, headers={
        "Ocp-Apim-Subscription-Key": env("AZURE_SPEECH_KEY"),
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "raw-16khz-16bit-mono-pcm",
        "User-Agent": "tts-compare",
    })
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return to_16k_mono(r.content, raw_sr=TARGET_SR)


CALLS = {
    "minimax": call_minimax,
    "aliyun_cosyvoice": call_aliyun_cosyvoice,
    "aliyun_qwen": call_aliyun_qwen,
    "volcengine": call_volcengine,
    "azure": call_azure,
}


# ---------------------------------------------------------------- billing estimate (list prices 2026-11-09)
def billable(text: str, vendor: str) -> int:
    """Billable chars by vendor rule: a Chinese char counts 2 for MiniMax / 百炼 / Azure, 1 for Volcengine."""
    han = sum(1 for c in text if "一" <= c <= "鿿" or "　" <= c <= "〿" or "＀" <= c <= "￯")
    return len(text) + (han if vendor in ("minimax", "aliyun_cosyvoice", "aliyun_qwen", "azure") else 0)


PRICE_PER_10K = {  # what one 10k billable characters costs at list price (see README for sources)
    "minimax": "¥3.5 (speech-2.8-hd, CN) / $1.0 (intl hd)",
    "aliyun_cosyvoice": "¥1 (cosyvoice-v3-flash, 北京)",
    "aliyun_qwen": "¥0.8 (qwen3-tts-flash, 北京)",
    "volcengine": "¥3 (豆包语音合成 2.0 后付费)",
    "azure": "$0.15 (Neural) / $0.22 (HD)",
}


# ---------------------------------------------------------------- driver
def load_sentences(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            sid, lang, cat, text = line.split("\t", 3)
            rows.append({"id": sid, "lang": lang, "category": cat, "text": text})
    return rows


def env_ok(vendor: str) -> bool:
    for spec in VENDORS[vendor]["env"]:
        alternatives = [alt.split("+") for alt in spec.split("|")]
        if not any(all(os.environ.get(n) for n in alt) for alt in alternatives):
            return False
    return True


def pick_voices(vendor: str, max_voices: int | None, only: list[str] | None) -> list[Voice]:
    voices = VENDORS[vendor]["voices"]
    if only:
        known = {v.id: v for v in voices}
        # unknown IDs are allowed (any stock voice from the vendor's list); treat as zh unless they look English
        guess = lambda x: "en" if x.lower().startswith(("en", "english", "loong")) else "zh"  # noqa: E731
        return [known.get(x) or Voice(x, x, guess(x)) for x in only]
    if max_voices:
        zh = [v for v in voices if v.lang == "zh"][:max_voices]
        en = [v for v in voices if v.lang == "en"][: max(1, (max_voices + 1) // 2)]
        return zh + en
    return voices


def run_vendor(vendor: str, sentences: list[dict], args) -> None:
    voices = pick_voices(vendor, args.max_voices, args.voices)
    jobs = [(v, s) for v in voices for s in sentences if v.lang == "zh" or s["lang"] == "en"]
    chars = sum(billable(s["text"], vendor) for _, s in jobs)
    print(f"[{vendor}] {len(voices)} voices, {len(jobs)} clips, ~{chars} billable chars "
          f"(price per 10k: {PRICE_PER_10K[vendor]})")
    if args.dry_run:
        return
    if not env_ok(vendor):
        print(f"[{vendor}] skipped: set {' and '.join(VENDORS[vendor]['env'])}")
        return
    out_dir = OUT_ROOT / vendor
    rows = []
    for n, (v, s) in enumerate(jobs, 1):
        path = out_dir / v.id.replace(" ", "_").replace("/", "_").replace(":", "_") / f"{s['id']}.wav"
        row = {"vendor": vendor, "voice": v.id, "voice_label": v.label, "voice_lang": v.lang, **s,
               "path": str(path), "sample_rate": TARGET_SR}
        if path.exists() and not args.force:
            row["duration_s"] = sf.info(str(path)).duration
            rows.append(row)
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                audio = CALLS[vendor](s["text"], v, s["lang"])
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")
                row.update(duration_s=len(audio) / TARGET_SR, synth_seconds=round(time.time() - t0, 2))
                row.pop("error", None)
                break
            except Exception as e:  # noqa: BLE001 - record any vendor failure in the manifest and go on
                row["error"] = f"{type(e).__name__}: {e}"
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5 * (attempt + 1))
                elif attempt == 0:
                    time.sleep(1)
                else:
                    break
        status = "ERR " + row["error"][:120] if "error" in row else f"{row['duration_s']:.1f}s"
        print(f"[{vendor}] {n}/{len(jobs)} {v.id} {s['id']} {status}")
        rows.append(row)
        time.sleep(args.sleep)
    out_dir.mkdir(parents=True, exist_ok=True)
    # merge with the existing manifest: a partial run (--only / --voices) must not drop the other rows
    # (2026-11-10: re-running one sentence left the listening page with 12 of 180 clips)
    mpath = out_dir / "manifest.jsonl"
    done = {(r["voice"], r["id"]) for r in rows}
    if mpath.exists():
        old = [r for r in map(json.loads, open(mpath, encoding="utf-8")) if (r["voice"], r["id"]) not in done]
        rows = old + rows
    with open(mpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{vendor}] wrote {out_dir / 'manifest.jsonl'} ({sum('error' in r for r in rows)} errors)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", required=True, choices=[*VENDORS, "all"])
    ap.add_argument("--voices", help="comma-separated stock voice IDs (overrides the default list)")
    ap.add_argument("--max-voices", type=int, help="first N zh voices (+ ceil(N/2) en voices) of the default list")
    ap.add_argument("--sentences", type=Path, default=SENTENCES)
    ap.add_argument("--only", help="comma-separated sentence ids, e.g. s01,s02")
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between requests (rate limits)")
    ap.add_argument("--force", action="store_true", help="re-synthesise existing wavs")
    ap.add_argument("--dry-run", action="store_true", help="only count clips / billable characters")
    args = ap.parse_args()
    args.voices = args.voices.split(",") if args.voices else None
    sentences = load_sentences(args.sentences)
    if args.only:
        keep = set(args.only.split(","))
        sentences = [s for s in sentences if s["id"] in keep]
    for vendor in VENDORS if args.vendor == "all" else [args.vendor]:
        try:
            run_vendor(vendor, sentences, args)
        except Exception as e:  # noqa: BLE001
            print(f"[{vendor}] failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
