"""Rule-based request analysis & task orchestration (README section 4/9).

Matches an ASR transcript against command patterns and returns which
authentication tier applies:
  - "open"          -> run immediately, no auth
  - "gated"         -> requires SV against a *claimed* user_id before running
  - "personalized"  -> requires SID (nearest centroid) before running

Kept deliberately simple/deterministic (substring + keyword matching) per
the README's rationale: predictable behavior matters more than flexibility
for a verification-gated demo. Swap for Intent+Entity or an LLM later by
replacing `match_intent` — nothing else in the pipeline needs to change.
"""
from dataclasses import dataclass
from enum import Enum

from src.config import settings


class CommandType(str, Enum):
    OPEN = "open"
    GATED = "gated"
    PERSONALIZED = "personalized"


@dataclass
class Intent:
    name: str
    command_type: CommandType


# Each pattern is a list of keywords; a transcript matches if ANY keyword
# in the list appears in the (lowercased) transcript. Vietnamese keywords
# shown here — extend freely per your actual use cases.
_COMMAND_PATTERNS: list[tuple[str, CommandType, list[str]]] = [
    ("get_weather", CommandType.OPEN, ["thời tiết", "trời hôm nay", "mưa không"]),
    ("get_time", CommandType.OPEN, ["mấy giờ", "thời gian bây giờ"]),
    ("read_last_message", CommandType.GATED, ["đọc tin nhắn", "tin nhắn mới nhất", "tin nhắn gần nhất"]),
    ("unlock_calendar", CommandType.GATED, ["mở lịch", "mở khóa lịch", "xem lịch riêng tư"]),
    ("play_my_music", CommandType.PERSONALIZED, ["phát nhạc", "mở nhạc của tôi", "bật nhạc"]),
    ("my_schedule", CommandType.PERSONALIZED, ["lịch của tôi", "lịch hôm nay của tôi"]),
]


def match_intent(transcript: str) -> Intent | None:
    """Returns the first matching Intent, or None if nothing matches
    (caller should treat this as settings.command_unknown)."""
    text = transcript.lower().strip()
    for name, command_type, keywords in _COMMAND_PATTERNS:
        if any(kw in text for kw in keywords):
            return Intent(name=name, command_type=command_type)
    return None
