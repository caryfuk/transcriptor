# transcriptor

Local transcription, speaker segmentation and summary for any audio file.
Everything runs on this machine: mlx-whisper (large-v3), pyannote, Ollama.

## Use

```sh
cp ~/Downloads/meeting.m4a sources/
uv run transcriptor                 # processes everything in sources/ not done yet
uv run transcriptor --watch         # or: keep running, pick up new files as they land
```

Results land in `output/<file name>/`:

| file | what |
|------|------|
| `transcript.txt` | plain text |
| `transcript.srt`, `.vtt` | subtitles with timestamps |
| `transcript_segmented.md` | timestamped blocks per speaker turn (paragraphs if `--no-diarize`) |
| `summary.md` | overview, key points, decisions, action items, open questions |
| `segments.json` | the segmented transcript as data |
| `speakers.json` | `SPEAKER_00 → name` map, edit it and run `--redo segment` |
| `raw/` | untouched Whisper output (word timestamps, confidences) and diarization |

## Options worth knowing

```sh
uv run transcriptor -l cs                         # fix the language instead of auto-detecting
uv run transcriptor -p "Alice, Bob, Kubernetes, OKR"        # names/terms Whisper should know
uv run transcriptor -n 4                          # exact number of speakers, if known
uv run transcriptor -c "weekly product sync"      # context for the summary
uv run transcriptor -s English                    # summary in another language
uv run transcriptor -m gemma4:12b-mlx             # different Ollama model (or TRANSCRIPTOR_MODEL)
uv run transcriptor --redo summary sources/x.m4a  # rerun a stage (and the ones after it)
uv run transcriptor --no-diarize --no-summary     # transcript only
```

Stages are `transcribe → diarize → segment → summary`. Each writes its result
before the next starts, so an interrupted run picks up where it stopped.

## Pipeline

1. **Normalize** — ffmpeg to 16 kHz mono, high-pass 70 Hz + EBU R128 loudnorm.
   On real meeting audio this transcribed 25–34% more words than the raw file:
   Whisper otherwise drops quiet, distant speakers. Diarization gets the
   *unnormalized* audio, since normalization hurts speaker embeddings.
2. **Transcribe** — mlx-whisper large-v3, word timestamps,
   `condition_on_previous_text=False` so one bad segment can't derail the rest.
3. **Diarize** — pyannote `speaker-diarization-community-1`; speaker assigned per
   *word*, since Whisper segments often straddle a speaker change.
4. **Summarize** — Ollama (default `qwen3.8:27b-mlx`, 32k context). Long
   recordings are summarized in ~40k-character parts, then merged.

## Setup (once)

```sh
brew install ffmpeg ollama
uv sync
ollama pull qwen3.8:27b-mlx
```

The Whisper and pyannote models are downloaded to the Hugging Face cache on the
first run. pyannote's model is gated: accept its terms on huggingface.co and run
once with `HF_TOKEN=...`; after that it loads from cache with no network.
