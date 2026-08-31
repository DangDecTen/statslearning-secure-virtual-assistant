# Configuration

This is the configuration for the virtual assistant. More details in the `src/config.py` while this file only cover the following topics:
- Speaker recognition models
- Verification/Identification threshold
- Enrollment threshold

## Choosing the model variant

`src/config.py` controls which ECAPA-TDNN embedding weights `SpeakerModel`
loads:

```python
# ===== Speaker verification / identification =====
spk_model_variant = "finetuned"  # "pretrained" | "finetuned"
```

- `"pretrained"`: stock `speechbrain/spkrec-ecapa-voxceleb`, no extra
  download.
- `"finetuned"` (default): loads the same base architecture, then
  downloads `best_checkpoint_rec98.pt` from the Hugging Face repo above and
  loads it into `embedding_model` only. The checkpoint is loaded with
  `torch.load(..., weights_only=True)` so it can't execute arbitrary code —
  see `src/speaker.py: SpeakerModel._load_finetuned_embedding_weights`.

To switch, either edit `spk_model_variant` in `config.py`, or pass
`--variant` to `tests/test_core.py` (see below) to override it for a single
run without touching the file.

**Important — re-enrollment required after switching variants.** Only the
L2-normalized centroid is stored per user (`src/db.py`); raw enrollment
audio and the variant that produced the centroid are not stored. Embeddings
from the pretrained and fine-tuned models are not directly comparable, so
switching `spk_model_variant` invalidates existing centroids for SV/SID
purposes — re-run `--enroll` for every user under the new variant before
trusting `--run` results.

## Choosing the verification/identification threshold

```python
spk_verify_threshold = 0.435  # minDCF threshold
```

The `spk_verify_threshold` is a cosine similarity (higher = stricter, range from -1 to 1).

- **SV** (`SpeakerModel.verify`): accepts the claim if
  `score >= spk_verify_threshold`, where `score` is the cosine similarity
  between the probe embedding and the claimed user's stored centroid.
- **SID** (`SpeakerModel.identify`): picks the enrolled user with the
  highest similarity to the probe, then still requires
  `best_score >= spk_verify_threshold` before accepting that match —
  otherwise it returns "unknown" rather than the closest (but insufficiently
  close) user.

How to choose a **decision threshold**:
- EER, minDCF thresholds
- Choose a threshold to satisfy a fixed FA or FR criterion (Neyman–Pearson);
- Varying threshold to find different FA/FR ratios and choosing one to give the desired FA/FR ratio.

## Choosing enrollment quality settings

Enrollment is about selecting good input material rather than making an
accept/reject decision on a live command:

```python
spk_enroll_threshold = 0.435  # currently use the same threshold as verify/identify

spk_min_enrollment = 3  # minimum utters to ask for user, use for user enrollment
spk_max_enrollment = 5  # maximum utters to ask for user, use as candidate utters
```
