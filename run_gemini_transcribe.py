"""Runner — benchmark Gemini 3.5 Transcribe / Transcribe Live on all eval languages.

API-based (no GPU), same LLM-track store as run_gemini.py. Biases each
language with its ISO 639-3 code (dataset keys are already ISO 639-3) so
small languages transcribe in-language instead of falling back to Igbo/Hausa.

Run:  python3 run_gemini_transcribe.py                        # sync model, all langs
      python3 run_gemini_transcribe.py --model gemini-3.5-transcribe-live
      python3 run_gemini_transcribe.py --langs twi_asante ewe dag
      python3 run_gemini_transcribe.py --max-workers 4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")

from benchmark.gemini_transcribe import (
    evaluate_transcribe, SYNC_MODEL, LIVE_MODEL,
)
from benchmark.evaluate import load_eval_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", help="ISO codes (default: all eval langs)")
    ap.add_argument("--model", default=SYNC_MODEL,
                    help=f"Gemini Transcribe model (default: {SYNC_MODEL})")
    ap.add_argument("--max-workers", type=int, help="concurrent requests")
    args = ap.parse_args()

    if args.model not in (SYNC_MODEL, LIVE_MODEL):
        sys.exit(f"Unknown model {args.model!r}; use {SYNC_MODEL} or {LIVE_MODEL}")

    langs = args.langs or list(load_eval_configs().keys())
    print("=" * 60)
    print(f"  Gemini Transcribe benchmark ({args.model}) · {len(langs)} languages")
    print("=" * 60)
    t0 = time.time()
    for i, iso in enumerate(langs, 1):
        print(f"\n===== [{i}/{len(langs)}] {iso} =====", flush=True)
        try:
            evaluate_transcribe(iso, model=args.model,
                                max_workers=args.max_workers)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
    print(f"\nDone in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()