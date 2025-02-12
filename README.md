# Fine Tuning Text History


[Fine Tune texts](https://edwarddonner.com/2024/01/11/fine-tune-llama-for-text-messages-part-1/)

## Setup

### Installation
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

### Config
Setting up a config file for tracking across runs
```yaml
model:
  base: meta-llama
  version: Meta-Llama-3.1-8B
  max_seq_length: 512
  use_fp16: true

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 5e-5
  weight_decay: 0.01
  eval_steps: 100
  eval_strategy: steps
  save_steps: 1000
  save_strategy: steps
  logging_steps: 50
  logging_strategy: steps

generation:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3

logging:
  use_wandb: true
  wandb_project: text_message_fine_tuning
  use_tensorboard: false
  tensorboard_dir: ./logs

data:
  folder: data
  tokenized: tokenized_output.json
  general: output.json
```



## Prepping Data

### Base Model
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


### Instruct Model
TBD

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




## Fine Tuning Model
In fine tuning the model for the text data, we want to use the base LLM model since we are just generating an endless text thread. If we wanted to fine tune a chatbot, we would use the Instruct model which does better with quesion/answer and task-driven commands.

Since we can't fine tune the model directly, we need to setup LoRA (Low Rank Adator) for the model to only train the attenstion layers.

```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers Q,V
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",  # No additional bias
    task_type="CAUSAL_LM"  # Language modeling task
)

# Prepare model for k-bit training (unfreezes LoRA adapters). This is needed if using quantization (bitsandbytes)
model = prepare_model_for_kbit_training(model)
```

And wrap the model with the config using PEFT (parameter efficient fine tuning)
```python
# Wrap the base model with LoRA adapters
model = get_peft_model(model, lora_config)
```

Once we have the model setup, we can setup the hugging face Trainer & TrainerArguments

```python
 training_args = TrainingArguments(
    output_dir="./fine-tuned-group-chat",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    
    save_steps=1000,
    logging_steps=50,  # Log progress every 50 steps
    logging_dir=logging_dir,  # Path for TensorBoard logs
    logging_strategy="steps",
    report_to=["tensorboard"],  # Report metrics to TensorBoard, launch with tensorboard --logdir ./logs
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)
```

And eventually start training
```python
torch.cuda.empty_cache() # Clear cache for GPU
model.train()
trainer.train()
```


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

For our case, since we used LoRA and PEFT, we need to load in that adaptor we saved and apply to the model

```python
from peft import PeftModel

fine_tuned_path = "./fine-tuned-model-path"  # Path to your fine-tuned model

model = AutoModelForCausalLM.from_pretrained(
            model_id_base,
            quantization_config=bnb_config, # bitsandbytes config used earlier
            device_map="auto"  # Automatically uses GPU, with CPU fallback if necessary
        )

tokenizer = AutoTokenizer.from_pretrained(model_id_base)

# Attach the fine-tuned LoRA adapters
fine_tuned_model = PeftModel.from_pretrained(model, fine_tuned_path)
```

We can then run inference thropugh our model for a new prompt

```python
model.eval() # Set model to evaluation 
max_seq_length = 256

def generate_response(prompt, max_length=max_seq_length, temperature=0.7, top_k=50):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate text
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,  # Enables sampling for more creative output
        pad_token_id=tokenizer.eos_token_id  # Prevents errors related to pad tokens
    )
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Inference
prompt = "Brayden: Hey everyone, what are we doing this weekend?"
response = generate_response(prompt)
print(response)
```


## Logging

### Weights and Biases
Needs to be installed
```bash
pip install wandb
```

And after logging in
```bash
wandb login
```

We can init in the code
```python
# Initialize W&B run
wandb.init(
    project=self.config["logging"]["wandb_project"],
    # name=self.run_name,
    config=self.config  # Pass your configuration
)
```