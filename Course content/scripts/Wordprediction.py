import os

from openai import OpenAI
#client = OpenAI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))


response = client.completions.create(
    model="gpt-3.5-turbo-instruct",   # or "davinci", etc.
    #prompt="My sister celebrates her birthday. Today I'd like to go to her",
    prompt="If we add two odd numbers, the sum wil be",
    max_tokens=1,  # only predict the next token
    logprobs=10    # return top 10 tokens with logprobs
)

print(response.choices[0].logprobs.top_logprobs[0])
response.choices[0].logprobs.top_logprobs[0][' a']


####################### long-span dependency

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model_name = "gpt2"  # or another causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Encode context
input_text = "There is a party today. Miriam celebrates her"
input_text = "Miriam passed away last week. Many people gather together today. There is a gathering, on the occasion of her"
input_text = "Miriam had her birthday last week. Many people gather together today. There is a gathering, on the occasion of her"
inputs = tokenizer(input_text, return_tensors="pt")

# Get logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits


# logits.shape
# torch.Size([1, 7, 50257])

# Focus on the last token’s distribution
last_token_logits = logits[0, -1, :]
probs = F.softmax(last_token_logits, dim=-1)

# Get top 10 predictions
topk = torch.topk(probs, k=10)
for idx, score in zip(topk.indices, topk.values):
    print(tokenizer.decode(idx), float(score))

 passing 0.15983469784259796
 birthday 0.10040120780467987
 death 0.06547503173351288
 50 0.025765499100089073
 30 0.02483692206442356
 70 0.022584792226552963
 20 0.02149866335093975
 funeral 0.019958702847361565
 40 0.01993800513446331
 25 0.017636623233556747

 birthday 0.3851492702960968
 20 0.025181438773870468
 birth 0.024490252137184143
 death 0.02277940697968006
 25 0.015567068941891193
 50 0.014186636544764042
 first 0.013026535511016846
 30 0.01267816498875618
 40 0.012569046579301357
 funeral 0.01083519496023655


###
model_name = "yhavinga/gpt2-large-dutch"  # or another causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
###
input_text = "Vandaag is het heel erg mooi"



 weer 0.8369532227516174
 en 0.02286016382277012
, 0.015040178783237934
 in 0.014925413765013218
 geweest 0.014597473666071892
. 0.011572600342333317
 om 0.006404077634215355
 zonnig 0.005224623251706362
 op 0.00437146844342351
 lente 0.004231028724461794

input_text = "Het meervoud van aardappel is"

 een 0.11991170048713684
 niet 0.05636293813586235
 in 0.04873490706086159
 de 0.032929427921772
 dus 0.02717394381761551
 het 0.025093471631407738
 aardappel 0.024663003161549568
   0.02350202202796936
 ook 0.022280486300587654
 hier 0.017818931490182877


##################### math reasoning

from transformers import pipeline
import torch

model_id =  "LLM360/K2-Think"

pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype="auto", # torch_dtype="auto", # deprecated
    device_map="auto"
) # this may take ages

messages = [
    {"role": "user", "content": "what is the next prime number after 2600?"},
]

outputs = pipe(
    messages,
    max_new_tokens=32768,
)
print(outputs[0]["generated_text"][-1])