"""Evaluation recipe for Qlerqly/griot-edge.

A 60.9M streaming Parallel-Conformer CTC model for Ghanaian languages (Akan,
Dagbani, Dagaare, Ewe, Fante, Ga, Ghanaian English). Like griot-nano-1 it does
not load through transformers: the repo ships its own `runtime.py`, and decoding
is the model card's default — 13-frame lookahead plus the bundled multilingual
KenLM (pyctcdecode beam 25, alpha 0.5, beta 1.0), exactly as `inference.py`
does. The model has no language conditioning, so one recipe serves every
language.

The repo's `inference.py` / `runtime.py` / `ctc_decode.py` are imported from the
downloaded model dir — they are the model author's code, not a dependency of
this benchmark. The benchmark calls `transcribe_batch(audio, ...)` with float
numpy arrays at the dataset sample rate.

Edit this file and open a PR to change how this model is evaluated.
"""

import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "Qlerqly/griot-edge"
CACHE_DIR = Path(os.environ.get("GRIOT_EDGE_DIR", "/mnt/volume_d2wey28/models/griot-edge"))

# Set to True to skip the bundled KenLM (greedy CTC). The default mirrors the
# model card's recommended decoding.
GREEDY = False


def _model_dir():
    if (CACHE_DIR / "model.safetensors").exists():
        return CACHE_DIR
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        MODEL, local_dir=str(CACHE_DIR),
        allow_patterns=["*.py", "*.json", "*.safetensors", "kenlm/*"],
        ignore_patterns=["benchmark-*", "deployment-verification.json"],
    ))


def _import_inference(model_dir):
    sys.path.insert(0, str(model_dir))
    spec = importlib.util.spec_from_file_location("griot_edge_inference", model_dir / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GriotEdge:
    def __init__(self, device="cpu"):
        model_dir = _model_dir()
        self.inf = _import_inference(model_dir)
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model, vocab, self.id_to_token = self.inf.load_model(model_dir, self.device, dtype)

        rc = json.loads((model_dir / "release_config.json").read_text())
        self.lookahead = rc.get("default_lookahead_frames")
        self.beam_width = rc.get("beam_width", 25)
        self.decoder = None
        if not GREEDY:
            from ctc_decode import build_kenlm_decoder
            self.decoder = build_kenlm_decoder(
                vocab, model_dir / rc["bundled_kenlm"],
                alpha=rc.get("lm_alpha", 0.5), beta=rc.get("lm_beta", 1.0),
            )

    @torch.inference_mode()
    def transcribe_batch(self, audio_arrays, sample_rate=16000, progress_cb=None):
        from _live import prepare_audio
        results = []
        for i, arr in enumerate(audio_arrays):
            try:
                audio = prepare_audio(np.asarray(arr), int(sample_rate))
                text = self.inf.transcribe(
                    self.model, self.id_to_token, audio, self.device,
                    decoder=self.decoder, beam_width=self.beam_width,
                    lookahead_frames=self.lookahead,
                )
            except Exception:
                # One bad clip must not lose the category; an empty hypothesis
                # is excluded from the score rather than penalised (see _score).
                text = ""
            results.append(text)
            if progress_cb and (i + 1) % 50 == 0:
                progress_cb(i + 1, len(audio_arrays))
        return results

    def cleanup(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_wrapper(device="cpu"):
    return GriotEdge(device=device)
