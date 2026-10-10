"""Who spoke when, with pyannote (runs locally once the model is in the HF cache)."""
import os, sys
from . import audio

PIPELINE = "pyannote/speaker-diarization-community-1"


def diarize(wav, num_speakers=None):
    import torch
    from pyannote.audio import Pipeline
    try:
        pipe = Pipeline.from_pretrained(PIPELINE, token=os.environ.get("HF_TOKEN"))
    except Exception:
        # gated model: without a token, fall back to the local cache
        os.environ["HF_HUB_OFFLINE"] = "1"
        pipe = Pipeline.from_pretrained(PIPELINE)
    if pipe is None:
        raise RuntimeError(f"cannot load {PIPELINE}: set HF_TOKEN once to download it")
    if torch.backends.mps.is_available():
        try:
            pipe.to(torch.device("mps"))
        except Exception as e:
            print(f"  mps unavailable ({e}), using cpu", file=sys.stderr)

    # pass the waveform directly: avoids pyannote's own decoder dependencies
    wf = torch.from_numpy(audio.load(wav)).unsqueeze(0)
    kw = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe({"waveform": wf, "sample_rate": 16000}, **kw)

    # exclusive = overlapping speech resolved to one speaker: right for labelling words
    return [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": spk}
            for s, _, spk in out.exclusive_speaker_diarization.itertracks(yield_label=True)]
