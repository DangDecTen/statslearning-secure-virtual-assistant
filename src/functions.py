"""Concrete implementations for the >=3 required use cases (README section 5).
Each takes an optional `user_id` (populated only for gated/personalized
intents, after SV/SID has already succeeded) and returns text for TTS.

These are stubs with realistic shapes — swap the bodies for real integrations
(a weather API, a message store, a music-preferences table, etc).
"""
from datetime import datetime


def get_weather(user_id: str | None = None) -> str:
    # Open function: no auth, same answer for anyone.
    return "Hôm nay trời nắng nhẹ, nhiệt độ khoảng hai mươi tám độ."


def get_time(user_id: str | None = None) -> str:
    now = datetime.now().strftime("%H:%M")
    return f"Bây giờ là {now}."


def read_last_message(user_id: str) -> str:
    # Gated: only reachable after SV succeeds for `user_id`.
    return f"Đây là tin nhắn gần nhất của {user_id}: (nội dung tin nhắn mẫu)."


def unlock_calendar(user_id: str) -> str:
    return f"Lịch riêng tư của {user_id} đã được mở khóa."


def play_my_music(user_id: str) -> str:
    # Personalized: `user_id` comes from SID (nearest centroid match).
    return f"Đang phát danh sách nhạc yêu thích của {user_id}."


def my_schedule(user_id: str) -> str:
    return f"Đây là lịch trình hôm nay của {user_id}: (lịch mẫu)."


# name -> callable, matched against Intent.name from orchestrator.py
FUNCTION_REGISTRY = {
    "get_weather": get_weather,
    "get_time": get_time,
    "read_last_message": read_last_message,
    "unlock_calendar": unlock_calendar,
    "play_my_music": play_my_music,
    "my_schedule": my_schedule,
}
