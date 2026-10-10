"""Summaries with a local Ollama model. Long transcripts are summarized in chunks, then merged."""
import json, os, re, sys, urllib.error, urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("TRANSCRIPTOR_MODEL", "qwen3.8:27b-mlx")
NUM_CTX = 32768
CHUNK_CHARS = 40000   # ~12k tokens of transcript; leaves room for prompt + output in NUM_CTX

SYSTEM = (
    "You summarize transcripts of recordings (meetings, calls, lectures, interviews). "
    "The transcript is machine-generated: expect misheard words and wrong punctuation; "
    "infer the intended meaning but never invent facts. "
    "Write in the language you are told to use."
)

FINAL = """Write a summary of this recording in Markdown with these sections (omit a section if it has no content):

## Overview
2-4 sentences: what this recording is, who takes part, the main purpose.

## Key points
Bullets of the substantive topics and what was said about them. Attribute to speakers when known.

## Decisions
What was agreed or decided, with any numbers, dates or vote results.

## Action items
Who does what, by when (if stated).

## Open questions
Unresolved issues or disagreements.

Reference timestamps like [00:12:34] for important points.
{extra}
---
{body}"""

PARTIAL = """This is part {i} of {n} of a longer transcript. Write detailed notes in Markdown bullets:
topics discussed, positions of each speaker, facts and numbers, decisions, action items.
Keep timestamps like [00:12:34] for important points. Do not add an introduction.
{extra}
---
{body}"""


def chat(prompt, model=MODEL, think=False):
    req = {"model": model, "stream": False, "think": think, "keep_alive": "2m",
           "options": {"num_ctx": NUM_CTX, "temperature": 0.3},
           "messages": [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]}
    try:
        r = _post(req)
    except urllib.error.HTTPError as e:
        if "think" not in e.read().decode(errors="replace"):
            raise
        req.pop("think")   # model without thinking support
        r = _post(req)
    return re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S).strip()


def _post(req):
    data = json.dumps(req).encode()
    with urllib.request.urlopen(urllib.request.Request(
            f"{HOST}/api/chat", data, {"Content-Type": "application/json"}), timeout=3600) as f:
        return json.load(f)


def check(model=MODEL):
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as f:
            names = {m["name"] for m in json.load(f)["models"]}
    except OSError as e:
        return f"ollama not reachable at {HOST} ({e}); start it with `ollama serve`"
    if model not in names and f"{model}:latest" not in names:
        return f"ollama model '{model}' not installed; `ollama pull {model}`"


def chunks(text, size=CHUNK_CHARS):
    """Split on block boundaries (blank lines) so no speaker turn is cut in half."""
    out, cur = [], ""
    for block in text.split("\n\n"):
        if cur and len(cur) + len(block) > size:
            out.append(cur)
            cur = ""
        cur += block + "\n\n"
    return out + [cur] if cur.strip() else out


def summarize(transcript, model=MODEL, context=None, think=False, language=None):
    extra = f"\nContext about this recording: {context}\n" if context else ""
    # instruction goes after the transcript: at the end of a long prompt it isn't lost
    tail = f"\n---\nWrite your answer in {language}, including the section headings." if language else \
        "\n---\nWrite your answer in the same language as the transcript."
    parts = chunks(transcript)
    if len(parts) == 1:
        return chat(FINAL.format(extra=extra, body=transcript) + tail, model, think)
    notes = []
    for i, p in enumerate(parts, 1):
        print(f"  summary: part {i}/{len(parts)}", file=sys.stderr, flush=True)
        notes.append(f"### Part {i}\n" + chat(PARTIAL.format(i=i, n=len(parts), extra=extra, body=p) + tail, model, think))
    print("  summary: merging", file=sys.stderr, flush=True)
    return chat(FINAL.format(extra=extra + "\nBelow are notes from consecutive parts of the recording.\n",
                             body="\n\n".join(notes)) + tail, model, think)


def language_name(code):
    if not code:
        return None
    from mlx_whisper.tokenizer import LANGUAGES
    return LANGUAGES.get(code, code).title()
