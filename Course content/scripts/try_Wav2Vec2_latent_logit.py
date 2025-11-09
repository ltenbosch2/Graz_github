# what I did before

pip install transformers
pip install torch
pip install soundfile

##### get transcription by Wav2Vec2

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf

# Load processor and ASR model (with decoder head)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file
file_path = "C:/Users/louis/OneDrive - Radboud Universiteit/Bureaublad/Graz/Course content/scripts/thisisanotherexample.wav"
speech, sr = sf.read(file_path)

# Convert to float32 and add batch dimension
input_values = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True).input_values

model.config.output_hidden_states = True

# Disable gradient tracking
with torch.no_grad():
    outputs = model(input_values)

# we now have the hidden layers and the logit tensor
logits = outputs.logits  # size batch, time stamp, token id
hidden_states = outputs.hidden_states
print(len(hidden_states))  # Number of layers

# Take argmax and decode to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print("TRANSCRIPTION:")
print(transcription[0])

# NOW THIS IS AN EXAMPLE OFFER OF A DOCUMENTS IN WHICH WE TRY TO PROF A NUMBER OF CONJECTURES AND THAT ARE ATTRIBUTED TO MISTER EXUY AN S AND IN THIS EPLE I WOULD LIKE TO TRY TO TO CONFITIU ABOUT THE VALIDITY OF THE PROFS THAT HAS BEEN A GIVEN BY A MISTER WY SO  LETS FIRST START WITH THE PROF GIVEN BY MISTN X AT WHICH SHOWS THAT IF A E IS AN ALIPTIC CURVE WITH A A CERTAIN DEGREE THEN WE HAVE THE FOLLOWING PROCEDURE


# how to see the vocab:
vocab = processor.tokenizer.get_vocab()
for token, index in vocab.items():
    print(index, token)

# how to see which token is recognized at which time stamp?
timestep = 57
logit_vector = logits[0, timestep]    # length = vocab size
predicted_token_id = torch.argmax(logit_vector).item()
token = processor.tokenizer.decode([predicted_token_id])
print(predicted_token_id, token)

# latent representations
# Example: take layer 6 representations
layer_index = 6
layer_6_repr = hidden_states[layer_index]   # shape: (batch, time_steps, hidden_dim)
print(layer_6_repr.shape)


## heat map

import matplotlib.pyplot as plt
import torch

# logits: [batch, time, vocab]
logits_orig = logits
logits = logits.squeeze(0).cpu()    # remove batch dim -> [time, vocab]

# Convert to numpy for plotting
logits_np = logits.numpy()

plt.figure(figsize=(10, 6))
plt.imshow(logits_np.T, aspect='auto')  # vocab x time
plt.title("Wav2Vec2 Logits Heatmap (CTC Output)")
plt.xlabel("Time (frames)")
plt.ylabel("Vocabulary token ID")
plt.colorbar()
plt.show()



################# from logits to probs ###################

....

# The logits tensor (before softmax)
logits = outputs.logits
print(logits.shape)  # [batch_size, sequence_length, vocab_size]

# If you like: convert values in logit tensor to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)



#### 
### here continue with own experiments (e.g. look at class separability in latent vector space for two different vowels as function of some layer k)