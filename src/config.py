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
    # "pretrained" -> stock speechbrain/spkrec-ecapa-voxceleb
    # "finetuned"  -> base architecture + fine-tuned embedding_model weights
    spk_model_variant = "finetuned"  # "pretrained" | "finetuned"

    spk_model_source = "speechbrain/spkrec-ecapa-voxceleb"
    spk_model_savedir = "data/pretrained_models/spkrec-ecapa-voxceleb"
    spk_embedding_dim = 192
    spk_device = "cpu"

    # Fine-tuned checkpoint (only used when spk_model_variant == "finetuned")
    spk_finetuned_repo_id = "Nampfiev1995/pvad-speechbrain-ft"
    spk_finetuned_filename = "best_checkpoint_rec98.pt"
    # Pin a specific commit hash once you've vetted the file, e.g. "a1b2c3d..."
    spk_finetuned_revision = None

    # decision threshold (verify, identify, enroll)
    spk_verify_threshold = 0.435
    spk_enroll_threshold = 0.435
    spk_min_enrollment = 3
    spk_max_enrollment = 5

    # ===== Orchestrator =====
    command_unknown = "unknown"

    # ===== TTS =====
    tts_engine = "gtts"  # "gtts" (primary per README) or "f5" later
    tts_speed = 1.0
    tts_gtts_lang = "vi"


settings = Settings()
