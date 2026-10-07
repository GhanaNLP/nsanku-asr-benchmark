"""Runner — benchmark Qlerqly/griot-edge on every language it supports.

The model card lists Akan, Dagbani, Dagaare, Ewe, Fante, Ga and Ghanaian
English. The HF language tags only declare ak/dag/dga/ee/en, so tag-driven
discovery would miss Fante and Ga; this runner names the eval languages
explicitly instead. Ghanaian English has no config in ghana-speech-eval.
Akan covers both Twi varieties in the eval set.

Runs on CPU (the model is 61M params and decoding is CPU-bound KenLM beam
search), as WORKERS processes each holding their own model + decoder.
Recipe: recipes/Qlerqly_griot-edge.py. Results land in benchmarks/{iso}.yaml.

Run:  python3 run_griot_edge.py
      python3 run_griot_edge.py --langs dag ewe --workers 12
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.config import NUM_SAMPLES
from benchmark.dataset import load_eval_samples
from benchmark.evaluate import (language_categories, load_eval_configs,
                                save_benchmark, save_transcriptions, _score)
from benchmark.recipes import load_recipe

MODEL_ID = "Qlerqly/griot-edge"
LANGS = ["twi_akuapem", "twi_asante", "dag", "dga", "ewe", "fat", "gaa"]

_wrapper = None


def _init_worker():
    global _wrapper
    import torch
    torch.set_num_threads(1)
    _wrapper = load_recipe(MODEL_ID).build_wrapper(device="cpu")


def _transcribe(item):
    audio, sr = item
    return _wrapper.transcribe_batch([audio], sample_rate=sr)[0]


def _build_result(per_category, wers, cers):
    wer = round(sum(wers) / len(wers), 4) if wers else None
    cer = round(sum(cers) / len(cers), 4) if cers else None
    r = {"model": MODEL_ID, "model_url": f"https://huggingface.co/{MODEL_ID}",
         "owner": "Qlerqly", "model_class": "non-llm", "params": "0.1B",
         "wer": wer, "cer": cer, "score": cer, "per_category": per_category,
         "source": "evaluated"}
    if wer is None:
        r["error"] = "no_valid_output"
    return r


def evaluate(iso, pool):
    cats = language_categories(iso)
    language = load_eval_configs()[iso]["language"]
    names = [c for c, _ in cats]
    print(f"\n===== {iso} ({language}) categories={names} =====", flush=True)
    per_category, wers, cers = {}, [], []
    for category, config in cats:
        samples = load_eval_samples(config, NUM_SAMPLES)
        if not samples:
            continue
        refs = [s["text"] for s in samples]
        items = [(s["audio"], s["sample_rate"]) for s in samples]
        audio_sec = sum(len(s["audio"]) / s["sample_rate"] for s in samples)
        print(f"  '{category}': {len(samples)} clips, {audio_sec / 60:.0f} min", flush=True)
        t0 = time.time()
        hyps = pool.map(_transcribe, items, chunksize=4)
        elapsed = time.time() - t0
        wer, cer, valid = _score(refs, hyps)
        save_transcriptions(iso, MODEL_ID, category, refs, hyps)
        per_category[category] = {
            "wer": round(wer, 4) if wer is not None else None,
            "cer": round(cer, 4) if cer is not None else None,
            "samples": len(samples), "valid": valid,
            "avg_seconds_per_sample": round(elapsed / len(samples), 2)}
        if wer is not None:
            wers.append(wer)
            cers.append(cer)
            print(f"    WER {wer:.2%}  CER {cer:.2%}  ({elapsed:.0f}s)", flush=True)
        save_benchmark(iso, language, names, [_build_result(per_category, wers, cers)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
    try:
        for iso in args.langs:
            evaluate(iso, pool)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
