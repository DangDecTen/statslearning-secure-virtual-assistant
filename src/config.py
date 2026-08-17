from pathlib import Path


class Settings:
    # ===== Paths =====
    data_dir = Path("data")
    db_path = data_dir / "assistant.db"
    tts_output_dir = data_dir / "tts_out"

    # ===== Audio =====
    sample_rate: int = 16000
    input_channels: int = 1

    # ===== ASR =====
    asr_model_name = "small"  # tiny, base, small, medium, large-v3
    asr_device = "cpu"
    asr_compute_type = "int8"  # good CPU speed/accuracy tradeoff for faster-whisper
    asr_language = "vi"

    # ===== Speaker verification / identification =====
    spk_model_source = "speechbrain/spkrec-ecapa-voxceleb"
    spk_model_savedir = "data/pretrained_models/spkrec-ecapa-voxceleb"
    spk_embedding_dim = 192
    # EER-derived threshold from Requirement 1 (re-tune on live mic audio).
    spk_verify_threshold = 0.50
    spk_min_enrollment_clips = 2
    spk_max_enrollment_clips = 3

    # ===== Orchestrator =====
    command_unknown = "unknown"

    # ===== TTS =====
    tts_engine = "gtts"  # "gtts" (primary per README) or "f5" later
    tts_speed = 1.0
    tts_gtts_lang = "vi"


settings = Settings()
