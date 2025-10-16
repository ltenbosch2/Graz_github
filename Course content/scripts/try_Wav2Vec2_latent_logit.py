# what I did before

pip install transformers
pip install torch
pip install soundfile

########## wav2vec, hidden layers and logits

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf

# Load pretrained Wav2Vec2 and processor processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Make sure hidden states are returned
model.config.output_hidden_states = True  

# get audio
## Example: audio tensor (1 sec of fake audio for demo) # You’d normally load with librosa/torchaudio 
# input_values = torch.randn(1, 16000)  # batch size 1, 16k samples


sample, sr = sf.read("C:/Users/louis/Downloads/thisisanotherexample.wav")               # this works

# input_values is a tensor # this is OK for the remaining code
# sample is an array, not OK for the rest of the code, make it a torch tensor
s = torch.from_numpy(sample)
input_values = s.unsqueeze(0)
input_values = DoubleToFloat(input_values)

# Forward pass
with torch.no_grad():
    outputs = model(input_values)

# `hidden_states` is a tuple with one entry per layer
#   - hidden_states[0] = embeddings before transformer
#   - hidden_states[1:] = hidden states after each layer
hidden_states = outputs.hidden_states  

# Example: take layer 6 representations
layer_index = 6
layer_6_repr = hidden_states[layer_index]   # shape: (batch, time_steps, hidden_dim)

print(layer_6_repr.shape)

### to see the logits do

# Load pretrained model + processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") in this model no logit

# Load audio (should be 16kHz mono for this model)
speech, rate = sf.read("C:/Users/louis/Downloads/thisisanotherexample.wav")
inputs = processor(speech, sampling_rate=rate, return_tensors="pt", padding=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# The logits tensor (before softmax)
logits = outputs.logits
print(logits.shape)  # [batch_size, sequence_length, vocab_size]

# Example: convert to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Example: greedy decoding
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)
print(transcription)