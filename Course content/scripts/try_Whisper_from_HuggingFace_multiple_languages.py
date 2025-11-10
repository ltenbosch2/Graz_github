# what I did before:

pip install PyTorch --use-pep517
pip install torch --use-pep517
pip torchcodec
pip install soundfile



##### Whisper from Huggingface (more freedom, no OpenAI API KEY required, but lagging behind compared to OpenAI version)

from transformers import WhisperProcessor, WhisperForConditionalGeneration # this might take some time
import soundfile as sf
import torch

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None # essential?

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

## load wav file directly
#sample, sr = sf.read("C:/Users/louis/Downloads/thisisanotherexample.wav")               # this works
sample, sr = sf.read("C:/Users/louis/OneDrive - Radboud Universiteit/Bureaublad/Graz/Course content/scripts/thisisanotherexample.wav") # here take your own directory
#input_features = processor(sample, sampling_rate=16000, return_tensors="pt").input_features 


inputs = processor(sample, sampling_rate=16000, return_tensors="pt")

# inputs contains input_features (and attention_mask)
input_features = inputs.input_features
# attention_mask = inputs.attention_mask  # this is what you pass to generate(), if present


# generate token ids
predicted_ids = model.generate(input_features) # ok if single wav file
# predicted_ids = model.generate(input_features, attention_mask=attention_mask)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(transcription)

# [" This is an example of a document in which we try to prove a number of conjectures that are attributed to Mr. X, Y and Z. In a sequel, I would like to try to convince you about the validity of the proofs that have been given by Mr. Y. Let's first start with the proofs given by Mr. Z."]




######################### Experiment B: translation (works)

# The following example demonstrates English audio to French transcription by setting the decoder ids appropriately.

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import soundfile as sf
import torch

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="german", task="transcribe") # you might change german into french here


## load streaming dataset and read first audio sample
#ds = load_dataset("common_voice", "fr", split="test", streaming=True)
#ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
#input_speech = next(iter(ds))["audio"]
#input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

#sample, sr = sf.read("C:/Users/louis/Downloads/thisisanotherexample.wav")               # this works
sample, sr = sf.read("C:/Users/louis/OneDrive - Radboud Universiteit/Bureaublad/Graz/Course content/scripts/thisisanotherexample.wav") # here take your own directory
input_features = processor(sample, sampling_rate=16000, return_tensors="pt").input_features 


# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

[' Ja, das ist ein Beispiel eines Dokuments, in dem wir die Zahl der Konjunktur, die Sie mit Mr. X, Y und Z betreffen. In diesem Fall möchte ich Sie versuchen, die Verletzung der Proben zu konfinieren, die von Mr. Y gegeben worden ist.']

[" C'est un exemple d'un document où on essaie de prouver un nombre de conjectures qui sont attribuées à Mr. X, Y et Z. Et après, je vais essayer de confier la vulnérabilité des prouves qui ont été données par Mr. Y."]




