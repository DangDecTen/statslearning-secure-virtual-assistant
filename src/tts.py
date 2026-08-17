"""Text-to-speech. gTTS is the primary engine (per README) — no local model
needed, but requires internet access. Kept behind a small interface so a
local engine (e.g. F5-TTS) can be swapped in later without touching callers.
"""
import time
from abc import ABC, abstractmethod
from pathlib import Path

from gtts import gTTS

from src.config import settings


class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, gen_text: str, speed: float | None = None) -> Path:
        raise NotImplementedError


class GTTSEngine(TTSEngine):
    def synthesize(self, gen_text: str, speed: float | None = None) -> Path:
        output_dir = Path(settings.tts_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if speed is None:
            speed = settings.tts_speed
        slow = speed < 1.0  # gTTS only exposes a slow/normal flag, not a continuous rate

        out_path = output_dir / f"gtts_{int(time.time() * 1000)}.mp3"
        gTTS(text=gen_text, lang=settings.tts_gtts_lang, slow=slow).save(str(out_path))
        return out_path


def get_tts_engine() -> TTSEngine:
    if settings.tts_engine == "gtts":
        return GTTSEngine()
    raise ValueError(f"Unknown tts_engine '{settings.tts_engine}'")
