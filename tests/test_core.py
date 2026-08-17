"""Manual integration test for the core pipeline (no web layer).

This is NOT a pytest suite — it's a runnable script that exercises the
full chain end-to-end against real audio files, since the interesting
failure modes here (ASR accuracy, embedding similarity, threshold
behavior) only show up with real models/audio, not mocks.

Usage:
    python -m tests.test_core --enroll data/samples/alice_1.wav data/samples/alice_2.wav --user alice
    python -m tests.test_core --run

What --run does:
    1. Prints DB state (who's enrolled).
    2. Runs a fixed set of sample commands through the full pipeline:
       - one OPEN command            (no auth expected)
       - one GATED command           (claimed_user_id supplied, SV runs)
       - one GATED command, wrong id (should be denied)
       - one PERSONALIZED command    (SID runs, no claimed id needed)
       - one unrecognized transcript (should map to "unknown")
    3. Prints transcript, intent, auth result, response text, and the
       path to the synthesized reply audio for each, so you can eyeball
       correctness and also literally listen to the .mp3 outputs.

You need real .wav sample files to point this at — dataset clips from
Requirement 1 work fine, or record a few with the Streamlit UI later.
Edit SAMPLES below to point at files that exist on your machine.
"""
import argparse
import sys
from pathlib import Path

from src import db
from src.pipeline import Pipeline

# ---- Edit these paths to point at real audio on your machine ----
SAMPLES = {
    "open_command": "data/samples/weather_query.wav",
    "gated_command": "data/samples/alice_read_message.wav",
    "personalized_command": "data/samples/alice_play_music.wav",
    "unknown_command": "data/samples/gibberish.wav",
}
ENROLLED_USER_FOR_TEST = "alice"
WRONG_CLAIMED_USER = "bob"


def _check_samples_exist() -> bool:
    missing = [p for p in SAMPLES.values() if not Path(p).exists()]
    if missing:
        print("Missing sample audio files, edit SAMPLES in this script to point at real files:")
        for p in missing:
            print(f"  - {p}")
        return False
    return True


def run_enroll(user_id: str, audio_paths: list[str]) -> None:
    for p in audio_paths:
        if not Path(p).exists():
            print(f"ERROR: enrollment file not found: {p}")
            sys.exit(1)

    print(f"Enrolling '{user_id}' with {len(audio_paths)} clip(s)...")
    pipeline = Pipeline()
    pipeline.enroll_user(user_id, audio_paths)
    print(f"Done. '{user_id}' is now enrolled.")


def _print_result(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"  transcript      : {result.transcript!r}")
    print(f"  intent          : {result.intent_name} ({result.command_type})")
    print(f"  auth_passed     : {result.auth_passed}")
    print(f"  resolved_user_id: {result.resolved_user_id}")
    print(f"  response_text   : {result.response_text!r}")
    print(f"  audio_out       : {result.audio_out_path}")


def run_all() -> None:
    print("Enrolled users:", list(db.get_all_centroids().keys()) or "(none)")
    if ENROLLED_USER_FOR_TEST not in db.get_all_centroids():
        print(
            f"\nWARNING: '{ENROLLED_USER_FOR_TEST}' is not enrolled yet. "
            f"Gated/personalized tests below will correctly fail auth, "
            f"but won't prove verification actually works.\n"
            f"Run with --enroll first, e.g.:\n"
            f"  python -m tests.test_core --enroll <clip1.wav> <clip2.wav> --user {ENROLLED_USER_FOR_TEST}\n"
        )

    if not _check_samples_exist():
        return

    pipeline = Pipeline()

    # 1. Open: no auth required, should succeed regardless of claimed_user_id.
    result = pipeline.process_command(SAMPLES["open_command"])
    _print_result("OPEN command", result)
    assert result.command_type == "open"
    assert result.auth_passed is True

    # 2. Gated, correct claimed identity: should pass SV and run the function.
    result = pipeline.process_command(
        SAMPLES["gated_command"], claimed_user_id=ENROLLED_USER_FOR_TEST
    )
    _print_result(f"GATED command (claimed_user_id={ENROLLED_USER_FOR_TEST})", result)
    assert result.command_type == "gated"

    # 3. Gated, wrong claimed identity: SV should reject even if audio is
    #    the enrolled user's, because the claim itself is wrong.
    result = pipeline.process_command(
        SAMPLES["gated_command"], claimed_user_id=WRONG_CLAIMED_USER
    )
    _print_result(f"GATED command (claimed_user_id={WRONG_CLAIMED_USER}, expect deny)", result)
    assert result.auth_passed is False

    # 4. Personalized: SID should resolve the speaker without an explicit claim.
    result = pipeline.process_command(SAMPLES["personalized_command"])
    _print_result("PERSONALIZED command", result)
    assert result.command_type == "personalized"

    # 5. Unrecognized transcript: should map to command_unknown, not crash.
    result = pipeline.process_command(SAMPLES["unknown_command"])
    _print_result("UNKNOWN command", result)
    assert result.intent_name == "unknown"

    print("\nAll checks passed (see printed results above for auth correctness).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enroll", nargs="+", metavar="AUDIO_WAV", help="Audio clips to enroll a user with.")
    parser.add_argument("--user", help="user_id to enroll (required with --enroll).")
    parser.add_argument("--run", action="store_true", help="Run the full sample-command pipeline test.")
    args = parser.parse_args()

    if args.enroll:
        if not args.user:
            parser.error("--enroll requires --user")
        run_enroll(args.user, args.enroll)

    if args.run:
        run_all()

    if not args.enroll and not args.run:
        parser.print_help()


if __name__ == "__main__":
    main()
