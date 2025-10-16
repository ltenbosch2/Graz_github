# first do:
# pip install openai

#########################

from openai import OpenAI  # for making OpenAI API calls
# import urllib  # for downloading example audio files
import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-own-key"))

def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript.text


wavfile = 'C:/Users/louis/Downloads/thisisanexampleandthisalso.wav';


# now compare:

transcribe(wavfile, prompt="")

transcribe(wavfile, prompt="This aaa is uuhmmmm an uuuhhh example.... and uuuh this  uhmmm is an ex example.")

# prompt variation shows an effect in case of hesitations.

