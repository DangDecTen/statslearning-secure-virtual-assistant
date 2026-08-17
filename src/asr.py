"""Speech-to-text via faster-whisper, running locally on CPU.

faster-whisper (CTranslate2 backend) is used instead of vanilla openai-whisper
per the README's tech-stack choice: comparable accuracy, materially faster
CPU inference, and int8 quantization support (see settings.asr_compute_type).
"""
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from src.config import settings


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_s: float


class ASR:
    def __init__(self) -> None:
        self._model = WhisperModel(
            settings.asr_model_name,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        segments, info = self._model.transcribe(
            str(audio_path),
            language=settings.asr_language,
            vad_filter=True,  # trims silence, reduces hallucinated text on short clips
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_s=info.duration,
        )


if __name__ == "__main__":
    import sys

    asr = ASR()
    result = asr.transcribe(sys.argv[1])
    print(f"[{result.language}, {result.duration_s:.1f}s] {result.text}")
