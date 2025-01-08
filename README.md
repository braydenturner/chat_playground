# Fine Tuning Text History


## Setup


Need a few libraries for this to work

```bash
pip install torch torchvision torchaudio # Pytorch
pip install transformers datasets accelerate # Huggingface libraries for interfacing with transformers, manipulating datasets, and training 
pip install bitsandbytes # Used for running larger models by using quantization
```


Detecting if cuda is being used

```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 3080 Ti
```



## Prepping Data
If fine tuning the models, they perform best in the structure like

```json
{"prompt": "Hey, how was your day?", "response": "Pretty good! How about you?"}
{"prompt": "What's the plan for the weekend?", "response": "Probably just relaxing at home."}
```

For the text data we want to generate a sliding window of context like

```json
{"messages": [
    {"role": "Alice", "content": "Hey, anyone free for lunch tomorrow?"},
    {"role": "Bob", "content": "Yeah, I'm free."},
    {"role": "Charlie", "content": "I can make it too."},
    {"role": "Dana", "content": "Let's do 1 PM."}
]}
{"messages": [
    {"role": "Bob", "content": "Yeah, I'm free."},
    {"role": "Charlie", "content": "I can make it too."},
    {"role": "Dana", "content": "Let's do 1 PM."},
    {"role": "Alice", "content": "Perfect! See you all then."}
]}
{"messages": [
    {"role": "Bob", "content": "Did you guys watch the game last night?"},
    {"role": "Charlie", "content": "Yes! It was intense."},
    {"role": "Dana", "content": "I missed it. Who won?"}
]}
```
to maintain the way the text flows. This can be achieved from

```python
def preprocess_long_chat(chat_history, window_size=5, overlap=2):
    chunks = []
    for i in range(0, len(chat_history), window_size - overlap):
        chunk = chat_history[i:i + window_size]
        if len(chunk) > 1:
            chunks.append({"messages": chunk})
    return chunks
```

## Models
Currently using the open source llama models, LLaMA 7B and under can fit comfortably in VRAM (12GB). Anything more would require lower precision or quantization.

Loading the models with no changes to model size

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3-2"  # Replace with actual model path if local
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
```

And here is if we want to use quantization

```python
model_name = "meta-llama/Llama-3-2"  # Replace with actual model path if local
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
            # Most necessary
            load_in_4bit=True, # Drastically reduces VRAm. 32 bit by default
            bnb_4bit_compute_dtype=compute_dtype, # Specifies the computation precision during forward passes
            
            # Optional (but reccomended)
            bnb_4bit_quant_type="nf4", # Use normalized float 4 for better accuracy
            bnb_4bit_use_double_quant=True, # Apply double quantization to reduce VRAM
            
            # Situational
            llm_int8_enable_fp32_cpu_offload=True # Offload to CPU if out of memory
        )


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with quantization and auto device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically uses GPU, with CPU fallback if necessary
```


For `device_map` vs `to("cuda")`, rule of thumb

* Models ≤ 7B ➝ Use .to("cuda") for simplicity. Faster
* Models > 7B ➝ Use device_map="auto" to handle VRAM overflow efficiently. More scalable. Flexible. Prevents crashing when running out of memory.


## Inference

Once the model weights are loaded, new prompts can be passed to the tokenizer and then passed to the model.


```python
def generate_response(prompt, max_length=100, temperature=0.7):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate text
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,  # Controls randomness (higher = more creative)
        top_p=0.9,  # Nucleus sampling (focus on top 90% probability mass)
        repetition_penalty=1.1,  # Penalize repetitive outputs
        do_sample=True  # Enable sampling for more diverse results
    )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Explain the significance of black holes in astrophysics."
response = generate_response(prompt)
print(response)
```


This is for the general for raw text generation. If we are passing this in to a chatbot like model tuned for chat, the `apply_chat_template` on the tokenizer is necessary

```python
prompt = "I have tomatoes, basil and cheese at home. What can I cook for dinner?\n"

def inference(prompt: str) -> str:
    
    messages = [
        {"role": "system", "content": "You are a chatbot that helps with recipes. Be as over the top and add extra information about the history of the recipe.",},
        {"role": "user", "content":prompt},
    ]
    
    # Tokenize the chat
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", # PyTorch tensor
        tokenize=True, # Tokenize, returns tokens instead of text
        add_special_tokens=False, # Avoid <eos> like tokens
        return_dict=True, # Returns dictionary that is used below to move to GPU
        )
    
    # Move to GPU
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}

    # Generate
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
       # Only return response, not input passed in
    input_size = inputs['input_ids'].size(1)
    output = outputs[0][input_size:]
    
    # Decode tokens in to words
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    
    return decoded_output


print(inference(prompt))
```



## Fine Tuning Model
In fine tuning the model for the text data, we want to use the base LLM model since we are just generating an endless text thread. If we wanted to fine tune a chatbot, we would use the Instruct model which does better with quesion/answer and task-driven commands,