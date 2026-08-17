"""Wires together the per-turn runtime flow described in README section 4:

    audio -> ASR -> orchestrator -> (SV | SID | nothing) -> function -> TTS

This module has no web/UI dependency — `app/backend` (FastAPI, built later)
will just call `Pipeline.process_command(...)`.
"""
from dataclasses import dataclass
from pathlib import Path

from src.asr import ASR
from src.config import settings
from src.functions import FUNCTION_REGISTRY
from src.orchestrator import CommandType, match_intent
from src.speaker import SpeakerModel
from src.tts import get_tts_engine


@dataclass
class CommandResult:
    transcript: str
    intent_name: str
    command_type: str
    auth_passed: bool
    resolved_user_id: str | None
    response_text: str
    audio_out_path: Path | None


class Pipeline:
    def __init__(self) -> None:
        # Load heavy models once; reuse across turns.
        self._asr = ASR()
        self._speaker = SpeakerModel()
        self._tts = get_tts_engine()

    def process_command(
        self,
        audio_path: str | Path,
        claimed_user_id: str | None = None,
    ) -> CommandResult:
        """
        Args:
            audio_path: recorded command audio (this turn's utterance).
            claimed_user_id: required only for GATED intents — the identity
                the user is claiming to be, to verify against (README open
                question: same command utterance is used for SV here, rather
                than a separate fixed enrollment-style phrase).
        """
        transcript = self._asr.transcribe(audio_path).text
        intent = match_intent(transcript)

        if intent is None:
            return CommandResult(
                transcript=transcript,
                intent_name=settings.command_unknown,
                command_type="none",
                auth_passed=False,
                resolved_user_id=None,
                response_text="Xin lỗi, tôi không hiểu yêu cầu của bạn.",
                audio_out_path=None,
            )

        resolved_user_id: str | None = None
        auth_passed = True

        if intent.command_type == CommandType.GATED:
            if not claimed_user_id:
                return self._deny(transcript, intent, reason="Vui lòng cho biết bạn là ai trước.")
            result = self._speaker.verify(claimed_user_id, audio_path)
            auth_passed = result.accepted
            if not auth_passed:
                return self._deny(transcript, intent, reason="Tôi không thể xác minh giọng nói của bạn.")
            resolved_user_id = claimed_user_id

        elif intent.command_type == CommandType.PERSONALIZED:
            result = self._speaker.identify(audio_path)
            resolved_user_id = result.user_id
            auth_passed = resolved_user_id is not None
            if not auth_passed:
                return self._deny(transcript, intent, reason="Tôi chưa nhận ra giọng nói của bạn.")

        # OPEN intents skip straight here with resolved_user_id=None.
        handler = FUNCTION_REGISTRY[intent.name]
        response_text = (
            handler(resolved_user_id) if resolved_user_id is not None else handler()
        )

        audio_out = self._tts.synthesize(response_text)

        return CommandResult(
            transcript=transcript,
            intent_name=intent.name,
            command_type=intent.command_type.value,
            auth_passed=auth_passed,
            resolved_user_id=resolved_user_id,
            response_text=response_text,
            audio_out_path=audio_out,
        )

    def _deny(self, transcript: str, intent, reason: str) -> CommandResult:
        audio_out = self._tts.synthesize(reason)
        return CommandResult(
            transcript=transcript,
            intent_name=intent.name,
            command_type=intent.command_type.value,
            auth_passed=False,
            resolved_user_id=None,
            response_text=reason,
            audio_out_path=audio_out,
        )

    def enroll_user(self, user_id: str, audio_paths: list[str | Path]) -> None:
        self._speaker.enroll(user_id, audio_paths)
