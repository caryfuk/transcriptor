"""transcriptor: drop audio into sources/, get transcript + segmented transcript + summary in output/.

Each stage writes its result to output/<name>/ and is skipped next time if present,
so an interrupted run resumes where it stopped. --redo STAGE reruns that stage and
everything after it.
"""
import argparse, json, pathlib, shutil, sys, time
from . import audio, segment

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif",
        ".mp4", ".mov", ".mkv", ".webm", ".amr", ".3gp"}
STAGES = ["transcribe", "diarize", "segment", "summary"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def process(src: pathlib.Path, args):
    d = args.output / src.stem
    raw, work = d / "raw", d / "work"
    raw.mkdir(parents=True, exist_ok=True)
    work.mkdir(exist_ok=True)
    f_asr, f_dia = raw / "transcript.json", raw / "diarization.json"
    f_seg, f_sum = d / "segments.json", d / "summary.md"

    redo = STAGES[STAGES.index(args.redo):] if args.redo else []
    for stage, f in zip(STAGES, [f_asr, f_dia, f_seg, f_sum]):
        if stage in redo:
            f.unlink(missing_ok=True)

    dur = audio.duration(src)
    log(f"\n== {src.name}" + (f"  ({dur / 60:.1f} min)" if dur else ""))
    t0 = time.time()

    if not f_asr.exists():
        log("  normalizing loudness")
        wav = audio.asr_wav(src, work)
        log(f"  transcribing (whisper, language={args.language or 'auto'})")
        from . import asr
        t = time.time()
        r = asr.transcribe(wav, args.language, args.prompt)
        r["_meta"] = {"source": src.name, "model": asr.MODEL, "elapsed_s": round(time.time() - t, 1)}
        _write_json(f_asr, r)
        log(f"  transcribed in {r['_meta']['elapsed_s'] / 60:.1f} min, language={r.get('language')}")
    whisper = json.loads(f_asr.read_text())

    turns = None
    if args.diarize:
        if not f_dia.exists():
            log("  diarizing (pyannote)")
            try:
                from . import diarize
                t = time.time()
                _write_json(f_dia, diarize.diarize(audio.dia_wav(src, work), args.speakers))
                log(f"  diarized in {(time.time() - t) / 60:.1f} min")
            except Exception as e:
                log(f"  ! diarization failed, continuing without speakers: {e}")
        if f_dia.exists():
            turns = json.loads(f_dia.read_text())

    # speakers.json maps SPEAKER_00 -> a real name; edit it and rerun with --redo segment
    f_names = d / "speakers.json"
    blocks = segment.build(whisper, turns)
    if turns and not f_names.exists():
        _write_json(f_names, {s: s for s in sorted({b["speaker"] for b in blocks})})
    names = json.loads(f_names.read_text()) if f_names.exists() else {}
    if not f_seg.exists():
        segment.write_outputs(d, whisper, blocks, names)
        _write_json(f_seg, blocks)
        speakers = len({b["speaker"] for b in blocks if b["speaker"]})
        log(f"  segmented: {len(blocks)} blocks" + (f", {speakers} speakers" if speakers else ""))

    if args.summary and not f_sum.exists():
        from . import summarize
        err = summarize.check(args.model)
        if err:
            log(f"  ! skipping summary: {err}")
        else:
            log(f"  summarizing ({args.model})")
            t = time.time()
            text = (d / "transcript_segmented.md").read_text(encoding="utf-8")
            lang = args.summary_language or summarize.language_name(whisper.get("language"))
            s = summarize.summarize(text, args.model, args.context, args.think, lang)
            f_sum.write_text(f"# {src.stem} — summary\n\n{s}\n", encoding="utf-8")
            log(f"  summarized in {(time.time() - t) / 60:.1f} min")

    if not args.keep_audio:
        shutil.rmtree(work, ignore_errors=True)
    log(f"  done in {(time.time() - t0) / 60:.1f} min -> {d.relative_to(ROOT) if d.is_relative_to(ROOT) else d}")


def _write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def pending(args):
    files = sorted(p for p in args.sources.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith("."))
    return [p for p in files if not (args.output / p.stem / "segments.json").exists()
            or (args.summary and not (args.output / p.stem / "summary.md").exists())]


def watch(args):
    log(f"watching {args.sources} (ctrl-c to stop)")
    sizes = {}
    while True:
        for p in pending(args):
            size = p.stat().st_size
            if sizes.get(p) == size:          # unchanged since last poll: copy finished
                run_one(p, args)
                sizes.pop(p, None)
            else:
                sizes[p] = size
        time.sleep(args.interval)


def run_one(p, args):
    try:
        process(p, args)
    except Exception as e:
        log(f"  ! failed: {e}")
        if not args.watch:
            raise


def main():
    ap = argparse.ArgumentParser(prog="transcriptor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=pathlib.Path,
                    help="audio files (default: everything in sources/ not yet processed)")
    ap.add_argument("-l", "--language", help="spoken language code, e.g. cs, en (default: auto-detect)")
    ap.add_argument("-p", "--prompt", help="whisper prompt: names and domain terms that appear in the recording")
    ap.add_argument("-n", "--speakers", type=int, help="exact number of speakers, if known")
    ap.add_argument("-c", "--context", help="what the recording is, to guide the summary")
    ap.add_argument("-m", "--model", default=None, help="ollama model for the summary")
    ap.add_argument("-s", "--summary-language", help="summary language, e.g. English (default: same as recording)")
    ap.add_argument("--think", action="store_true", help="let a reasoning model think before summarizing (slower)")
    ap.add_argument("--no-diarize", dest="diarize", action="store_false", help="skip speaker detection")
    ap.add_argument("--no-summary", dest="summary", action="store_false", help="skip the ollama summary")
    ap.add_argument("--redo", choices=STAGES, help="rerun this stage and the ones after it")
    ap.add_argument("--keep-audio", action="store_true", help="keep the normalized WAVs in output/<name>/work/")
    ap.add_argument("-w", "--watch", action="store_true", help="keep running and process new files as they appear")
    ap.add_argument("--interval", type=float, default=10, help="watch poll interval in seconds")
    ap.add_argument("--sources", type=pathlib.Path, default=ROOT / "sources")
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "output")
    args = ap.parse_args()
    if args.model is None:
        from .summarize import MODEL
        args.model = MODEL
    args.sources.mkdir(exist_ok=True)
    args.output.mkdir(exist_ok=True)

    if args.watch:
        try:
            watch(args)
        except KeyboardInterrupt:
            pass
        return
    files = args.files or (sorted(p for p in args.sources.iterdir() if p.suffix.lower() in EXTS)
                           if args.redo else pending(args))
    if not files:
        log(f"nothing to do: put audio files into {args.sources}")
    for p in files:
        run_one(p.resolve(), args)


if __name__ == "__main__":
    main()
