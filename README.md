# Secure Virtual Assistant

- **Implemented** (`src/`): config, SQLite storage, ASR (faster-whisper), speaker verification/identification (ECAPA-TDNN via speechbrain), rule-based orchestrator, example use-case functions, TTS (gTTS), and a `Pipeline` class wiring all of it into the full per-turn runtime flow.
- **Not yet implemented**: `app/backend` (FastAPI), `app/frontend` (Streamlit). The pipeline is designed so the FastAPI layer will be a thin wrapper around `Pipeline.process_command()` / `Pipeline.enroll_user()` with no changes needed underneath.
- **Testing**: `tests/test_core.py` is a manual integration-test script (not pytest) that enrolls a user and runs sample commands through all three intent types against real audio. See section 10.

## 1. Setup

```bash
# On Windows, SpeechBrain need to run as administrator (or Developer Mode
# turned on) to run correctly with symlink. Then open terminal and VS Code

# Create conda env, then install ffmpeg first because it may confict with
# existing packages when installed later. Avoid using later versions in
# which python>3.12 were used.
conda install -c conda-forge ffmpeg==5.1.2
ffmpeg -version     # Check installation, hope it does not broke :(((

# To use SpeechBrain, support python>=3.9;<=3.12
conda install python=3.11
python --version
pip --version       # Hope nothing broke :(((

# Other
pip install -r requirements.txt
pip check           # Hope nothing broke :(((

# Install k2 (for SpeechBrain) using torch version that match our requirements.txt
# Link: https://k2-fsa.github.io/k2/index.html
pip install k2==1.24.4.dev20260625+cpu.torch2.4.1 -f https://k2-fsa.github.io/k2/cpu.html

# Tips: To avoid broken environment, gradually picking a working version
#       of the most important module (e.g. ffmpeg, speechbrain, torch,
#       etc.), then try different versions of the least important ones.
```

## 2. Running The Test

### 2.1 Note on first run

The first time `SpeakerModel()` or `ASR()` is constructed, SpeechBrain and
faster-whisper will download their pretrained weights automatically into
`data/pretrained_models/` and the faster-whisper cache respectively. No
manual step needed, but expect the first run to be slower and to require
network access.

### 2.2 Get sample audio

Create `data/samples/` and add `.wav` files. You need, at minimum:

- One clip with an "open" command (e.g. asking for the weather)
- Two enrollment clips for a test user (e.g. "alice")
  - One clip of that user speaking a "gated" command (e.g. "read my last message")
  - One clip of that user speaking a "personalized" command (e.g. "play my music")
- One clip of unrelated/gibberish speech

```text
open_weather:
    Cho tôi biết thời tiết hôm nay

gated_message:
    Xem tin nhắn mới nhất của tôi

personal_music:
    Hãy mở nhạc của tôi

unknown_command:
    Cả hai bên hãy cố gắng hiểu cho nhau
```

Edit the `SAMPLES` dict, `ENROLLED_USER_FOR_TEST`, and `WRONG_CLAIMED_USER`
constants near the top of `tests/test_core.py` to point at your actual files
and match your orchestrator's phrasing (see `src/orchestrator.py` for the
patterns it matches).

### 2.3 Enroll a user

```bash
python -m tests.test_core --enroll data/samples/alice_1.wav data/samples/alice_2.wav --user alice

python -m tests.test_core --enroll data/samples/bob_1.wav data/samples/bob_2.wav --user bob
```

Default model is configured in `src/config.py`. Or you can specify the model `--variant` here:

```bash
python -m tests.test_core --variant finetuned --enroll data/samples/alice_1.wav data/samples/alice_2.wav --user alice
```

### 2.4 Run the full pipeline test

```bash
python -m tests.test_core --run
```

or, matching whichever variant you enrolled under:

```bash
python -m tests.test_core --variant finetuned --run
```

This will:

1. Print which speaker-model variant is active and who's currently enrolled
   (and warn if the active variant may not match how existing users were
   enrolled — see §2).
2. Run five fixed scenarios through `Pipeline.process_command(...)`: an open
   command, a correctly-claimed gated command, a wrongly-claimed gated
   command (should be denied), a personalized (SID) command, and an
   unrecognized transcript.
3. Print the transcript, matched intent, auth result, response text, and
   the path to the synthesized reply `.mp3` for each, so you can both read
   the output and listen to it.

You can see a complete test run in `tests/README.md` for referecences.

## 3. Speaker Recognition Models

`src/config.py` controls which ECAPA-TDNN embedding weights `SpeakerModel`
loads:

```python
# ===== Speaker verification / identification =====
spk_model_variant = "pretrained"  # "pretrained" | "finetuned"

spk_model_source = "speechbrain/spkrec-ecapa-voxceleb"
spk_model_savedir = "data/pretrained_models/spkrec-ecapa-voxceleb"
spk_embedding_dim = 192

spk_finetuned_repo_id = "Nampfiev1995/pvad-speechbrain-ft"
spk_finetuned_filename = "best_checkpoint_rec98.pt"
spk_finetuned_revision = None  # pin a commit hash once vetted
```

- `"pretrained"` (default): stock `speechbrain/spkrec-ecapa-voxceleb`, no
  extra download.
- `"finetuned"`: loads the same base architecture, then downloads
  `best_checkpoint_rec98.pt` from the Hugging Face repo above and loads it
  into `embedding_model` only. The checkpoint is loaded with
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

**Re-tune the threshold after switching.** `spk_verify_threshold` (currently
`0.50`) was derived from the pretrained-model EER comparison in
`report.md`. A fine-tuned embedding space will likely shift the score
distribution — re-run your threshold sweep before relying on the fine-tuned
variant in anything resembling production.

## 4. Architecture & Data Flow

See details in `app/README.md` for product architecture and data flows.

## 5. Repository Structure

```
secure-virtual-assistant/
├── README.md
├── report.md                      # Requirement 1 report (dataset/model/training/eval)
├── requirements.txt
├── data/
│   ├── tts_out/                   # synthesized reply audio (gTTS output)
│   ├── pretrained_models/         # speechbrain ECAPA-TDNN checkpoint (auto-downloaded)
│   ├── samples/                   # <- create this; put your test .wav files here
│   └── assistant.db               # SQLite: enrolled users' centroids (created on first enroll)
├── app/
│   ├── backend/                   # FastAPI app (not yet built)
│   └── frontend/                  # Streamlit UI (not yet built)
├── src/
│   ├── config.py                  # central settings, incl. speaker-model variant toggle
│   ├── db.py
│   ├── asr.py
│   ├── speaker.py
│   ├── orchestrator.py
│   ├── functions.py
│   ├── tts.py
│   └── pipeline.py
├── tests/
│   ├── test_core.py               # manual integration test — run with real audio
│   └── README.md
└── notebooks/
```

`data/` (minus committed `.gitkeep`s) should stay out of version control —
it holds downloaded model weights, the local SQLite DB, and generated audio.

## 6. Open Tasks

- `app/backend` and `app/frontend` are not implemented — `Pipeline` is
  designed to be called directly from a FastAPI route later without changes.
- The DB schema (`src/db.py`) doesn't record which speaker-model variant
  produced a stored centroid. If you routinely swap variants, consider
  adding a `model_variant` column and having `verify`/`identify` check it.
- `spk_verify_threshold` needs re-tuning per model variant and ideally per
  deployment microphone/environment (see `report.md`).
