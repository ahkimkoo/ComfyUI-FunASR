"""
ComfyUI FunASR Speech-to-Text Node
Uses FunASR for offline Chinese speech recognition with VAD and punctuation.
Models are auto-downloaded from ModelScope on first run.
"""

import os
import folder_paths
import numpy as np
import torch
import torchaudio
import logging

logger = logging.getLogger("ComfyUI-FunASR")

# ── Model paths (under ComfyUI/models/funasr/) ──────────────────────────

FUNASR_MODEL_DIR = os.path.join(folder_paths.models_dir, "funasr")

MODELS = {
    "asr": {
        "model_id": "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "local_dir": os.path.join(FUNASR_MODEL_DIR, "paraformer-zh"),
    },
    "vad": {
        "model_id": "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "local_dir": os.path.join(FUNASR_MODEL_DIR, "fsmn-vad"),
    },
    "punc": {
        "model_id": "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
        "local_dir": os.path.join(FUNASR_MODEL_DIR, "ct-punc"),
    },
}


def _ensure_model(key: str) -> str:
    """Return local_dir; download from ModelScope if missing."""
    info = MODELS[key]
    local_dir = info["local_dir"]
    # Check if the model directory already has model files
    if os.path.isdir(local_dir) and any(
        f.endswith((".bin", ".onnx", ".pt", ".yaml", ".json"))
        for f in os.listdir(local_dir)
    ):
        return local_dir

    logger.info(f"[FunASR] Downloading {info['model_id']} → {local_dir} ...")
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(info["model_id"], local_dir=local_dir)
    logger.info(f"[FunASR] Download complete: {local_dir}")
    return local_dir


def _get_funasr_model():
    """Lazy-load the FunASR AutoModel (cached on the function object)."""
    if hasattr(_get_funasr_model, "_cache"):
        return _get_funasr_model._cache

    from funasr import AutoModel

    asr_dir = _ensure_model("asr")
    vad_dir = _ensure_model("vad")
    punc_dir = _ensure_model("punc")

    model = AutoModel(
        model=asr_dir,
        vad_model=vad_dir,
        punc_model=punc_dir,
        model_revision="v2.0.4",
        vad_model_revision="v2.0.4",
        punc_model_revision="v2.0.4",
    )
    _get_funasr_model._cache = model
    return model


# ── Audio loading helper ─────────────────────────────────────────────────

def _load_audio_to_16k_array(audio_path: str) -> np.ndarray:
    """Load any audio file and resample to 16 kHz mono float32 numpy array."""
    waveform, sample_rate = torchaudio.load(audio_path)
    # to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # resample to 16 kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


# ── ComfyUI Node ────────────────────────────────────────────────────────

class FunASRSpeechToText:
    """Transcribe Chinese speech to text using FunASR (offline, no internet needed at runtime)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "hotword": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional hotwords to boost recognition accuracy, one per line",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"
    OUTPUT_NODE = True

    def transcribe(self, audio: dict, hotword: str = ""):
        # audio dict from ComfyUI: {"waveform": Tensor, "sample_rate": int}
        waveform: torch.Tensor = audio["waveform"]  # (batch, channels, samples)
        sample_rate: int = audio["sample_rate"]

        # Flatten to mono numpy
        wav = waveform.squeeze(0)  # remove batch → (channels, samples)
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        wav_np = wav.cpu().numpy().astype(np.float32)

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            wav_t = torch.from_numpy(wav_np).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            wav_np = resampler(wav_t).squeeze(0).numpy()

        logger.info(f"[FunASR] Transcribing {len(wav_np)/16000:.1f}s of audio ...")
        model = _get_funasr_model()

        kwargs = {"input": wav_np, "batch_size_s": 300}
        if hotword.strip():
            kwargs["hotword"] = hotword.strip()

        result = model.generate(**kwargs)

        # Extract text from result
        texts = []
        for item in result:
            text = item.get("text", "")
            if text:
                texts.append(text)
        final_text = "".join(texts)

        logger.info(f"[FunASR] Result: {final_text[:200]}{'...' if len(final_text)>200 else ''}")
        return {"ui": {"text": [final_text]}, "result": (final_text,)}


class FunASRSpeechToTextFile:
    """Transcribe an audio file to text using FunASR."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and os.path.splitext(f)[1].lower() in audio_extensions
        ]
        return {
            "required": {
                "audio_file": (sorted(files), {"audio_upload": True}),
            },
            "optional": {
                "hotword": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional hotwords to boost recognition accuracy, one per line",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    CATEGORY = "FunASR"
    INPUT_NODE = True
    OUTPUT_NODE = True

    def transcribe(self, audio_file: str, hotword: str = ""):
        audio_path = folder_paths.get_annotated_filepath(audio_file)
        wav_np = _load_audio_to_16k_array(audio_path)

        logger.info(f"[FunASR] Transcribing {len(wav_np)/16000:.1f}s from {audio_file} ...")
        model = _get_funasr_model()

        kwargs = {"input": wav_np, "batch_size_s": 300}
        if hotword.strip():
            kwargs["hotword"] = hotword.strip()

        result = model.generate(**kwargs)

        texts = []
        for item in result:
            text = item.get("text", "")
            if text:
                texts.append(text)
        final_text = "".join(texts)

        logger.info(f"[FunASR] Result: {final_text[:200]}{'...' if len(final_text)>200 else ''}")
        return {"ui": {"text": [final_text]}, "result": (final_text,)}


# ── Node registration ───────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FunASRSpeechToText": FunASRSpeechToText,
    "FunASRSpeechToTextFile": FunASRSpeechToTextFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunASRSpeechToText": "FunASR Speech to Text",
    "FunASRSpeechToTextFile": "FunASR Speech to Text (File)",
}
