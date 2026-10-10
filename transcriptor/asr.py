"""Speech-to-text with mlx-whisper (Apple Silicon, fully local)."""
import gc

MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe(wav, language=None, prompt=None, model=MODEL):
    import mlx_whisper
    r = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=model, language=language, task="transcribe",
        best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6,
        condition_on_previous_text=False,     # stops one bad segment derailing the rest
        initial_prompt=prompt, word_timestamps=True,
        hallucination_silence_threshold=2.0, verbose=False,
    )
    _release()
    return r


def _release():
    """Drop the cached model so pyannote/ollama get the memory back."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder
    ModelHolder.model = ModelHolder.model_path = None
    gc.collect()
    mx.clear_cache()
