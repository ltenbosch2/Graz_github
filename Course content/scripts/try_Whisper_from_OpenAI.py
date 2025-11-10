# following https://cookbook.openai.com/examples/whisper_prompting_guide


# pip install openai

from openai import OpenAI  # for making OpenAI API calls
# import urllib  # for downloading example audio files
import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-QG4zJMzZE-MVviF4a_EHN2B19hCaFjJlGGSeNSKGy5iCTAGReU6gpgo988MBunYGPaZSUEaDZRT3BlbkFJ0WCGGxRcr_-eIIMkY.....   (here your own API key)"))

def transcribe(audio_filepath, prompt: str) -> str:
    """Given a prompt, transcribe the audio file."""
    transcript = client.audio.transcriptions.create(
        file=open(audio_filepath, "rb"),
        model="whisper-1",
        prompt=prompt,
    )
    return transcript.text


#example

wavfile = 'C:/Users/louis/Downloads/thisisanexampleandthisalso.wav';
wavfile = "C:/Users/louis/OneDrive - Radboud Universiteit/Bureaublad/Graz\Course content/scripts/thisisanexampleandthisalso.wav"
# reference: this is aaa this is aaa an example this is an example and this is also aaa an example
transcribe(wavfile, prompt="")
'This is an example, this is an example, and this is also an example.'

transcribe(wavfile, prompt="uhm aaa this is an example uhm aaa this uhm")
'this is aaa this is aaa an example this is an example and this is also aaa an example'

transcribe(wavfile, prompt="ici radio tour de france. je ne parle pas francais et toi?")
"C'est un exemple, c'est un exemple et c'est aussi un exemple."

transcribe(wavfile, prompt="Es soll nicht der letzte Besuch gewesen sein, sagt Pistorius. Denn er hat militärpolitisch einiges vor in Island. Ihm geht es vor allem um die militärische Schifffahrt.")
'Das ist ein Beispiel, das ist ein Beispiel und das ist auch ein Beispiel.'




# other examples
wavfile = 'C:/Users/louis/Downloads/thisisanotherexample.wav';
# "Yeah This is an example of a of a document in which we try to prove a number of conjectures that are attributed to Mr. X, Y, and Z. And in a sequel, I would like to try to to convince you about the validity of the proofs that have been given by Mr. Y. Soooo, let's first start with the proof given by Mr. X, which shows that if E is an elliptic curve with a certain degree, then we have the following procedure."

transcribe(wavfile, prompt = "")
"This is an example of a document in which we try to prove a number of conjectures that are attributed to Mr. X, Y and Z. And in the sequel, I would like to try to convince you about the validity of the proofs that have been given by Mr. Y. So, let's first start with the proof given by Mr. X, which shows that if E is an elliptic curve with a certain degree, then we have the following procedure."

transcribe(wavfile, prompt = "aaa aaa  this aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa aaa")
"This is an example of a document in which we try to prove a number of conjectures that are attributed to Mr. X, Y and Z and in a sequel I would like to try to convince you about the validity of the proofs that have been given by Mr. Y. So let's first start with the proof given by Mr. X which shows that if E is an elliptic curve with a certain degree then we have the following procedure."

transcribe(wavfile, prompt="Es soll nicht der letzte Besuch gewesen sein, sagt Pistorius. Denn er hat militärpolitisch einiges vor in Island. Ihm geht es vor allem um die militärische Schifffahrt.")
'Das hier ist ein Beispiel eines Dokuments, in dem wir versuchen zu beweisen, wie viele Konjunkturen zu den Mr. X, Y und Z angegeben sind. Und in der Sequenz möchte ich versuchen, Ihnen zu überzeugen, über die Qualität der Beweise, die von Mr. Y gegeben wurden. Lass uns zuerst mit dem Beweis von Mr. X beginnen, welcher zeigt, dass wenn E eine elliptische Kurve ist, mit einem bestimmten Grad, dann haben wir folgende Prozedur.'


#example
wavfile = 'C:/Users/louis/Downloads/nine_unclear.wav';
# nine
transcribe(wavfile, prompt = "")
"Noon."


wavfile = 'C:/Users/louis/Downloads/nine_unclear.wav';
transcribe(wavfile, prompt = "One two three four five size seven eight ")
'noon'
transcribe(wavfile, prompt = "The following word is an English digit ")
'Nun'
transcribe(wavfile, prompt = "The following word is an English digit with correct spelling")
'Null'

transcribe(wavfile, prompt = "One two three four five size seven eight. One two three four five size seven eight. The following word is the name of an English digit")



########################## the following doesn't run on my laptop (Python stops)
### Via HuggingFace:

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps=False
)
wavfile = 'C:/Users/louis/Downloads/thisisanexampleandthisalso.wav';

result = pipe(wavfile)
print(result["text"])

