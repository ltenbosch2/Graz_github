# following https://cookbook.openai.com/examples/whisper_prompting_guide


# pip install openai

from openai import OpenAI  # for making OpenAI API calls
# import urllib  # for downloading example audio files
import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript.text


wavfile = 'C:/Users/louis/Downloads/thisisanotherexample.wav';
wavfile = 'C:/Users/louis/Downloads/thisisanexample.wav';
wavfile = 'C:/Users/louis/Downloads/thisisanexampleandthisalso.wav';

# reference:
this is aaa this is aaa an example this is an example and this is also aaa an example

transcribe(wavfile, prompt="")
#'This is an example, this is an example, and this is also an example.'
transcribe(wavfile, prompt="This aaa is uuhmmmm an uuuhhh example.... and uuuh this  uhmmm is an ex example.")

# this works: prompt variation shows an effect in case of hesitations.

########################## the following doesn't run on my laptop (Python stops)

Via HuggingFace:

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps=False
)
wavfile = 'C:/Users/louis/Downloads/thisisanexampleandthisalso.wav';

result = pipe(wavfile)
print(result["text"])

