"""Turn raw Whisper output (+ optional diarization) into readable segments and files."""
import re

NOISE = re.compile(r"^[\s.…]*$")   # '' and '...' mark silence, not speech


def ts(x, sep=","):
    h, rem = divmod(int(x), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{int((x % 1) * 1000):03d}"


def short_ts(x):
    return ts(x, ".")[:8]


def clean_segments(whisper):
    return [s for s in whisper["segments"] if not NOISE.match(s["text"].strip())]


def _speaker_lookup(turns, bucket=10.0):
    idx = {}
    for t in turns:
        for b in range(int(t["start"] // bucket), int(t["end"] // bucket) + 1):
            idx.setdefault(b, []).append(t)

    def at(a, b):
        best, best_ov = None, 0.0
        for k in range(int(a // bucket), int(b // bucket) + 1):
            for t in idx.get(k, ()):
                ov = min(b, t["end"]) - max(a, t["start"])
                if ov > best_ov:
                    best, best_ov = t["speaker"], ov
        return best
    return at


def build(whisper, turns=None, gap=3.0, max_chars=1200):
    """Group Whisper segments into blocks.

    With diarization: one block per speaker turn. Speaker is voted per *word* by
    overlap, because Whisper segments often straddle a speaker change.
    Without: paragraphs split on pauses longer than `gap` seconds.
    """
    at = _speaker_lookup(turns) if turns else None
    blocks = []
    for s in clean_segments(whisper):
        spk = None
        if at:
            votes = {}
            for w in s.get("words") or []:
                sp = at(w["start"], w["end"])
                if sp:
                    votes[sp] = votes.get(sp, 0) + (w["end"] - w["start"])
            spk = max(votes, key=votes.get) if votes else (at(s["start"], s["end"]) or "UNKNOWN")
        text = s["text"].strip()
        prev = blocks[-1] if blocks else None
        same = prev and s["start"] - prev["end"] < gap and (
            prev["speaker"] == spk if at else len(prev["text"]) < max_chars)
        if same:
            prev["text"] += " " + text
            prev["end"] = s["end"]
        else:
            blocks.append({"start": s["start"], "end": s["end"], "speaker": spk, "text": text})
    return blocks


def write_outputs(d, whisper, blocks, names):
    segs = clean_segments(whisper)
    (d / "transcript.txt").write_text("\n".join(s["text"].strip() for s in segs) + "\n", encoding="utf-8")
    with open(d / "transcript.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text'].strip()}\n\n")
    with open(d / "transcript.vtt", "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{ts(s['start'], '.')} --> {ts(s['end'], '.')}\n{s['text'].strip()}\n\n")
    (d / "transcript_segmented.md").write_text(render(blocks, names, d.name), encoding="utf-8")


def render(blocks, names, title):
    out = [f"# {title}\n"]
    for b in blocks:
        who = f" **{names.get(b['speaker'], b['speaker'])}**" if b["speaker"] else ""
        out.append(f"[{short_ts(b['start'])}]{who}\n{b['text']}\n")
    return "\n".join(out)
