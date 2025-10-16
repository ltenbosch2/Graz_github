# what I did before:

pip install PyTorch --use-pep517
pip install torch --use-pep517
pip torchcodec
pip install soundfile



##### try Whisper from Huggingface


from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
sample, sr = sf.read("C:/Users/louis/Downloads/thisisanotherexample.wav")               # this works
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


