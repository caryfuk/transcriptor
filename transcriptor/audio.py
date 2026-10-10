"""Decode any input into the two 16 kHz mono WAVs the models want."""
import subprocess, wave, pathlib
import numpy as np

# High-pass removes rumble; EBU R128 loudnorm lifts quiet/distant speakers.
# Measured on real meeting audio: +25-34% words transcribed vs raw, same accuracy.
ASR_FILTER = "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11"


def _ffmpeg(src, dst, af=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    if af:
        cmd += ["-af", af]
    tmp = dst.with_suffix(".part.wav")
    subprocess.run(cmd + [str(tmp)], check=True)
    tmp.rename(dst)


def asr_wav(src, work: pathlib.Path):
    """Loudness-normalized: for Whisper."""
    p = work / "asr.wav"
    if not p.exists():
        _ffmpeg(src, p, ASR_FILTER)
    return p


def dia_wav(src, work: pathlib.Path):
    """Unprocessed: pyannote wants natural dynamics, normalization hurts speaker embeddings."""
    p = work / "dia.wav"
    if not p.exists():
        _ffmpeg(src, p)
    return p


def load(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def duration(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None
