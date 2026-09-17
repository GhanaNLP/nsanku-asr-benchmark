"""LLM ASR track — Gemini 3.5 Transcribe / Transcribe Live (dedicated STT).

Google's dedicated speech-to-text models (`gemini-3.5-transcribe` and the
streaming `gemini-3.5-transcribe-live`) transcribe audio WITHOUT a text prompt;
the transcript comes back in a typed `audio_transcription` part rather than as
plain text. This module scores them per category and writes to
benchmarks_llm/{iso}.yaml, the same LLM-track store the Gemini/Gemma LLMs use
so both can be merged into benchmarks/ afterwards.

Language is biased with the ISO 639-3 code of each eval language (the dataset
is keyed by these codes, e.g. 'ewe', 'dag', 'xsm'). The API's `language_codes`
field is what makes a small-language clip transcribe in that language instead
of silently falling back to Igbo/Hausa; auto-detection alone is unreliable for
the tonal languages in this eval set, and many of them have no BCP-47 code.

Models are recorded under `google/gemini-3.5-transcribe` and
`google/gemini-3.5-transcribe-live`.
"""

import collections
import os
import re
import time
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from .config import NUM_SAMPLES, ROOT
from .dataset import load_eval_samples
from .evaluate import (
    language_categories, load_eval_configs, save_transcriptions, _score,
)
from .gemini import (
    LLM_BENCHMARK_DIR, _encode_wav, default_prompt,
)

# Load .env file for GEMINI_API_KEY / HF_TOKEN
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

SYNC_MODEL = "gemini-3.5-transcribe"
LIVE_MODEL = "gemini-3.5-transcribe-live"
MODEL_OWNER = "google"

MAX_WORKERS = 8
MAX_RETRIES = 4
RETRY_BACKOFF_CAP = 20.0

# ISO 639-1 fallbacks for languages where the eval's 639-3 key is not what the
# STT model knows (Twi macro-language).
ISO_HINT_FALLBACK = {
    "twi_akuapem": "twi",
    "twi_asante": "twi",
}


def iso_hint(iso_code):
    """Return the language-code hint to bias transcription for an eval language.

    Defaults to the eval key itself (which is the ISO 639-3 code for almost all
    languages in this dataset); a small override map handles the rest.
    """
    return ISO_HINT_FALLBACK.get(iso_code, iso_code)


def _get_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


_failures = collections.Counter()
_failures_lock = threading.Lock()


def _record_failure(reason):
    with _failures_lock:
        _failures[reason] += 1


def reset_failures():
    with _failures_lock:
        _failures.clear()


def failure_summary():
    with _failures_lock:
        return dict(_failures)


def _part_text(parts):
    """Pull the transcript out of an `audio_transcription` part (fallback text)."""
    for p in parts:
        at = getattr(p, "audio_transcription", None)
        if at is not None and at.text:
            return at.text.strip()
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            return t.strip()
    return None


def _make_sync_transcribe(model, language_codes):
    """Transcribe one clip via generate_content (synchronous API)."""
    from google.genai import types

    def _transcribe(wav_bytes, prompt=None):
        client = _get_client()
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(
                        data=wav_bytes, mime_type="audio/wav")],
                    config=types.GenerateContentConfig(
                        audio_transcription_config=types.AudioTranscriptionConfig(
                            language_codes=list(language_codes) or None,
                        ),
                    ),
                )
                text = _part_text(resp.candidates[0].content.parts)
                if text:
                    return text
                _record_failure("empty response")
                return None
            except Exception as e:
                _record_failure(type(e).__name__)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(min(2.0 ** attempt, RETRY_BACKOFF_CAP))
        return None

    return _transcribe


def _transcribe_task(args):
    idx, wav_bytes, _, transcribe = args
    res = transcribe(wav_bytes)
    return idx, res or ""


def _build_live_transcribe(model, language_codes):
    """Transcribe clip-by-clip through the Live (streaming) API.

    Each clip opens its own Live session, streams the audio as raw PCM at
    roughly real-time pace, then collects the final `input_transcription`.
    A fresh session per clip is what keeps the benchmark's independence: no
    state leaks across clips, and a hung session can only cost that one clip.
    """
    import asyncio
    import numpy as np
    from google.genai import types

    session_timeout_s = 120.0

    async def _transcribe_async(arr, sample_rate):
        client = _get_client()
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=list(language_codes) or None,
            ),
        )
        async with client.aio.live.connect(model=model, config=config) as session:
            if session.setup_complete is None:
                _record_failure("live:no setup_complete")
                return None
            pcm = (np.clip(arr, -1, 1) * 32767).astype(np.int16).tobytes()
            chunk = int(sample_rate * 0.5) * 2  # 500 ms chunks
            for i in range(0, len(pcm), chunk):
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm[i:i + chunk],
                        mime_type=f"audio/pcm;rate={sample_rate}",
                    ))
                await asyncio.sleep(0.02)
            await session.send_realtime_input(audio_stream_end=True)
            finals = []
            rx_task = asyncio.create_task(_receive_finals(session, finals))
            try:
                await asyncio.wait_for(rx_task, timeout=session_timeout_s)
            except asyncio.TimeoutError:
                rx_task.cancel()
            except Exception as e:
                _record_failure(f"live:{type(e).__name__}")
            return " ".join(finals) if finals else None

    def _transcribe(wav_bytes, prompt=None):
        import io
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(wav_bytes))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        try:
            return asyncio.run(_transcribe_async(arr, int(sr)))
        except Exception as e:
            _record_failure(f"live:{type(e).__name__}")
            return None

    return _transcribe


async def _receive_finals(session, finals):
    """Collect finalized `input_transcription` messages from a Live session."""
    async for resp in session.receive():
        sc = getattr(resp, "server_content", None)
        if sc is None:
            continue
        fin = getattr(sc, "input_transcription", None)
        if fin and fin.text:
            finals.append(fin.text.strip())


def _has_result(iso_code, model_id):
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if not path.exists():
        return False
    d = yaml.safe_load(open(path)) or {}
    want = {c for c, _ in language_categories(iso_code)}
    for b in d.get("benchmarks", []):
        if b.get("model") != model_id:
            continue
        return want <= set(b.get("per_category") or {})
    return False


def _save(iso_code, language, category_names, result):
    LLM_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    base = {}
    if path.exists():
        base = yaml.safe_load(open(path)) or {}
    by_model = {b["model"]: b for b in base.get("benchmarks", [])}
    by_model[result["model"]] = result
    out = {
        "iso_639_3": iso_code,
        "language": language,
        "num_samples_per_category": NUM_SAMPLES,
        "categories": category_names,
        "benchmarks": list(by_model.values()),
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)


def evaluate_transcribe(iso_code, model=SYNC_MODEL, max_workers=None):
    """Score a Gemini Transcribe model on one language across its categories.

    model         — 'gemini-3.5-transcribe' (sync) or
                    'gemini-3.5-transcribe-live' (streaming).
    max_workers   — concurrency; the sync model parallelises well, the Live
                    model is per-clip serial inside its own asyncio loop.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required. Make sure it is set or in .env")

    workers = max_workers or MAX_WORKERS
    model_id = f"google/{model}"
    model_url = f"https://ai.google.dev/gemini-api/docs/models#{model}"

    cats = language_categories(iso_code)
    if not cats:
        print(f"  {iso_code} not in eval set - skipping")
        return
    if _has_result(iso_code, model_id):
        print(f"  {model_id} already done for {iso_code} - skipping")
        return

    meta = load_eval_configs()[iso_code]
    language = meta["language"]
    category_names = [c for c, _ in cats]
    hint = iso_hint(iso_code)
    print(f"\n{'=' * 60}\n  {model_id} - {iso_code} ({language})  "
          f"hint={hint}  categories={category_names}\n{'=' * 60}", flush=True)

    if model == LIVE_MODEL:
        transcribe = _build_live_transcribe(model, [hint])
    else:
        transcribe = _make_sync_transcribe(model, [hint])

    existing = {}
    path = LLM_BENCHMARK_DIR / f"{iso_code}.yaml"
    if path.exists():
        d = yaml.safe_load(open(path)) or {}
        for b in d.get("benchmarks", []):
            if b.get("model") == model_id:
                existing = b.get("per_category") or {}

    per_category = dict(existing)
    cat_wers, cat_cers = [], []
    for category, config in cats:
        if category in per_category and per_category[category].get("wer") is not None:
            print(f"  Category '{category}' already done - skipping", flush=True)
            cat_wers.append(per_category[category]["wer"])
            cat_cers.append(per_category[category]["cer"])
            continue

        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        print(f"  Category '{category}' ({len(samples)} samples, "
              f"{workers} workers)...", flush=True)

        tasks = []
        for i, s in enumerate(samples):
            wav_bytes = _encode_wav(s["audio"], s["sample_rate"])
            tasks.append((i, wav_bytes, "", transcribe))

        hyps = [""] * len(samples)
        done_count = 0
        reset_failures()
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_transcribe_task, t): t for t in tasks}
            for future in as_completed(futures):
                idx, hyp = future.result()
                hyps[idx] = hyp
                done_count += 1
                if done_count % 50 == 0 or done_count == len(samples):
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed else 0
                    eta = (len(samples) - done_count) / rate if rate else 0
                    print(f"      {done_count}/{len(samples)}  "
                          f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

        elapsed = time.time() - t0
        rate = len(samples) / elapsed if elapsed else 0
        print(f"      Done {len(samples)} samples in {elapsed:.0f}s "
              f"({rate:.1f}/s)", flush=True)

        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso_code, model_id, category, refs, hyps)
        failures = failure_summary()
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples),
            "valid": valid,
            "avg_seconds_per_sample": round(elapsed / max(len(samples), 1), 2),
            **({"failures": failures} if failures else {}),
        }
        if failures:
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(failures.items()))
            print(f"    {len(samples) - valid} clip(s) unscored — {detail}",
                  flush=True)
        if wer is not None:
            cat_wers.append(wer)
            cat_cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  (valid {valid}/{len(samples)})",
                  flush=True)

        avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
        avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
        result = {
            "model": model_id,
            "model_url": model_url,
            "owner": MODEL_OWNER,
            "model_class": "llm",
            "params": "API",
            "language_code_hint": hint,
            "wer": avg_wer,
            "cer": avg_cer,
            "per_category": per_category,
            "source": "evaluated",
        }
        if avg_wer is None:
            result["error"] = "no_valid_output"
        _save(iso_code, language, category_names, result)

    avg_wer = round(sum(cat_wers) / len(cat_wers), 4) if cat_wers else None
    avg_cer = round(sum(cat_cers) / len(cat_cers), 4) if cat_cers else None
    result = {
        "model": model_id,
        "model_url": model_url,
        "owner": MODEL_OWNER,
        "model_class": "llm",
        "params": "API",
        "language_code_hint": hint,
        "wer": avg_wer,
        "cer": avg_cer,
        "per_category": per_category,
        "source": "evaluated",
    }
    if avg_wer is None:
        result["error"] = "no_valid_output"
    _save(iso_code, language, category_names, result)
    if avg_wer is not None:
        print(f"  FINAL (avg of {len(cat_wers)} categories): "
              f"WER {avg_wer:.2%}  CER {avg_cer:.2%}", flush=True)
    return result