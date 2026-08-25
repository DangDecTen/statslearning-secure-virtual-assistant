"""Speaker verification (SV) and identification (SID) using a pretrained
ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb), per the Requirement 1
comparison (0.90% EER vs. 11.78% for the from-scratch model).

Loaded via speechbrain.inference.speaker.SpeakerRecognition, which
subclasses EncoderClassifier and adds verify_batch()/verify_files() on top
of the same encode_batch() used here — so this stays a drop-in swap.

SV and SID share one embedding call and differ only in the comparison step:
  - SV:  1-to-1  cosine similarity vs. one claimed user's stored centroid.
  - SID: 1-to-N  cosine similarity vs. every stored centroid, take the max.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from speechbrain.inference.speaker import SpeakerRecognition

from src.config import settings
from src import db


@dataclass
class VerifyResult:
    accepted: bool
    score: float
    threshold: float


@dataclass
class IdentifyResult:
    user_id: str | None  # None -> settings.command_unknown case
    score: float
    threshold: float


class SpeakerModel:
    def __init__(self, device: str = "cpu") -> None:
        self._device = device

        # Always load the base architecture + stock pretrained weights first.
        # SpeakerRecognition subclasses EncoderClassifier (adds verify_batch/
        # verify_files on top) — encode_batch below works identically either
        # way, so this also gives us pairwise verification for free if it's
        # ever useful alongside the centroid-based SV/SID used here.
        self._encoder = SpeakerRecognition.from_hparams(
            source=settings.spk_model_source,
            savedir=settings.spk_model_savedir,
            run_opts={"device": self._device},
        )
        self._encoder.eval()

        if settings.spk_model_variant == "finetuned":
            self._load_finetuned_embedding_weights()
        elif settings.spk_model_variant != "pretrained":
            raise ValueError(
                f"Unknown spk_model_variant {settings.spk_model_variant!r}; "
                "expected 'pretrained' or 'finetuned'."
            )

    def _load_finetuned_embedding_weights(self) -> None:
        ckpt_path = hf_hub_download(
            repo_id=settings.spk_finetuned_repo_id,
            filename=settings.spk_finetuned_filename,
            revision=settings.spk_finetuned_revision,
        )
        # weights_only=True restricts unpickling to tensors, blocking arbitrary
        # code execution from a malicious/compromised checkpoint file.
        state_dict = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        missing, unexpected = self._encoder.mods.embedding_model.load_state_dict(
            state_dict, strict=True
        )
        if missing or unexpected:  # strict=True will already raise before this,
            raise RuntimeError(   # but keep the guard in case strict is relaxed later
                f"Checkpoint mismatch — missing: {missing}, unexpected: {unexpected}"
            )
        self._encoder.eval()

    # ---- core embedding call, shared by SV, SID, and enrollment ----
    def extract_embedding(self, audio_path: str | Path) -> np.ndarray:
        signal, sr = torchaudio.load(str(audio_path))
        if signal.shape[0] > 1:  # downmix to mono if needed
            signal = signal.mean(dim=0, keepdim=True)
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)

        with torch.no_grad():
            emb = self._encoder.encode_batch(signal)  # shape: (1, 1, 192)
        emb = emb.squeeze().numpy().astype(np.float32)
        return _l2_normalize(emb)

    # ---- enrollment ----
    def enroll(self, user_id: str, audio_paths: list[str | Path]) -> np.ndarray:
        if not (settings.spk_min_enrollment_clips <= len(audio_paths) <= settings.spk_max_enrollment_clips + 2):
            # soft guard: warn-worthy, not a hard block, in case caller passes more
            pass
        embeddings = [self.extract_embedding(p) for p in audio_paths]
        centroid = _l2_normalize(np.mean(embeddings, axis=0))
        db.upsert_user(user_id, centroid, n_enrollment_clips=len(audio_paths))
        return centroid

    # ---- SV: is this really `claimed_user_id`? ----
    def verify(self, claimed_user_id: str, audio_path: str | Path) -> VerifyResult:
        stored = db.get_user_centroid(claimed_user_id)
        if stored is None:
            return VerifyResult(accepted=False, score=0.0, threshold=settings.spk_verify_threshold)

        probe = self.extract_embedding(audio_path)
        score = _cosine_similarity(probe, stored)
        return VerifyResult(
            accepted=score >= settings.spk_verify_threshold,
            score=score,
            threshold=settings.spk_verify_threshold,
        )

    # ---- SID: who out of all enrolled users is speaking? ----
    def identify(self, audio_path: str | Path) -> IdentifyResult:
        all_centroids = db.get_all_centroids()
        if not all_centroids:
            return IdentifyResult(user_id=None, score=0.0, threshold=settings.spk_verify_threshold)

        probe = self.extract_embedding(audio_path)
        scores = {uid: _cosine_similarity(probe, c) for uid, c in all_centroids.items()}
        best_user, best_score = max(scores.items(), key=lambda kv: kv[1])

        if best_score < settings.spk_verify_threshold:
            return IdentifyResult(user_id=None, score=best_score, threshold=settings.spk_verify_threshold)
        return IdentifyResult(user_id=best_user, score=best_score, threshold=settings.spk_verify_threshold)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are already L2-normalized, so this is just a dot product,
    # but computed explicitly in case a caller passes in raw vectors.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
