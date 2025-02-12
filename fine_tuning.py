import os
import json
import torch
import yaml
import wandb

import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from config import config
from transformers import AutoTokenizer, EarlyStoppingCallback, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback
from transformers.integrations import TensorBoardCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from evaluate import load
from datetime import datetime


token = config.APIkeys.HFToken


class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()



class FineTunedChat:
    
    
    def __init__(self, config_file = "model_config.yaml"):
        
        self.config = self._load_config(config_file)
        self.run_name = self._generate_run()
        self._initalize_wandb()
        
        if self.config["model"]["use_quantization"]:
            self.bnb_config = self._bits_and_bytes()
        else:
            self.bnb_config = None
        
        
        self.model_name = f"{self.config["model"]["base"]}/{self.config["model"]["version"]}"
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained()
        

    def _load_config(self, file_name):
        # Load YAML configuration
        with open(f"config/{file_name}", "r") as file:
            config = yaml.safe_load(file)
            
            return config
        
    def _generate_run(self):
        
        
        dir = "fine-tuned-group-chat/"
        try:
            # List all items in the directory
            items = os.listdir(dir)

            # Filter for folders only
            folders = [item for item in items if os.path.isdir(os.path.join(dir, item))]

            num_prev_runs = len(folders)
        except FileNotFoundError:
            print(f"Directory not found: {dir}")
            num_prev_runs = 0
        
        
        run_name = f"run_{num_prev_runs + 1}"
        
        return run_name
        
    def _initalize_wandb(self):
        
        # Initialize W&B run
        wandb.init(
            project=self.config["logging"]["wandb_project"],
            # name=self.run_name,
            config=self.config  # Pass your configuration
        )
            
            
    def _bits_and_bytes(self):
        return BitsAndBytesConfig(
            load_in_4bit=True, # 8 bit uses too much memory
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
    def process_data(self, reset = False):
        """
        Stream the chat data from JSONL file, apply sliding window processing, and tokenize.
        Uses Hugging Face datasets streaming to prevent memory overload.
        """
        
        tokenized_output_file = f'{self.config["data"]["folder"]}/{self.config["data"]["tokenized"]}'
        output_file = f'{self.config["data"]["folder"]}/{self.config["data"]["general"]}'
        
        if os.path.exists(tokenized_output_file) and os.path.exists(output_file):
            if reset:
                os.remove(tokenized_output_file)
                os.remove(output_file)   
            else:
                print("Files already exist")
                return
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS token for padding
            
        dataset = load_dataset(
            "json",
            data_files={"train": "data/krusty_krab.jsonl"},
            split="train",
            streaming=True  # Enables streaming mode
        )
        
        # Sliding window parameters
        window_size = 5   # Number of messages per window
        overlap = 2       # Number of overlapping messages between windows

        max_seq_length = self.config["model"]["max_seq_length"]  # Reduce from 512 if possible
        
        # Persistent buffer to accumulate messages across stream batches
        persistent_buffer = []  
        
        def sliding_window(chat_history):
            """
            Process chat history into sliding window chunks with overlap.
            This helps maintain conversation context over multiple messages.

            Args:
                chat_history (list): List of chat messages (JSON objects).
            
            Returns:
                list: List of overlapping chunks of messages, each formatted for tokenization.
            """
            nonlocal persistent_buffer
            chunks = []
            
            for message in chat_history:
                # Format message as "<name>: <text>"
                speaker = message['name']
                text = message['text']
                persistent_buffer.append(f"{speaker}: {text}")
                
                # When enough messages accumulate, create a sliding window chunk
                if len(persistent_buffer) >= window_size:
                    # Take the first 'window_size' messages from the buffer
                    chunk = persistent_buffer[:window_size]
                    chunks.append({"messages": chunk})
                    
                    # Keep the last 'overlap' messages to ensure continuity across windows
                    persistent_buffer = persistent_buffer[-overlap:]
            
            return chunks
        
        def tokenize_and_save(chunks):
            """
            Tokenize overlapping chat chunks and save the results to disk in JSON format.
            Handles Hugging Face tokenizer's BatchEncoding object by converting tensors to lists.

            Args:
                chunks (list): List of chat message chunks to be tokenized.
                output_file (str): Path to save the tokenized output.
            """
            tokenized_data = []
            data = []
            
            for chunk in chunks:
                # Join chunk messages into a single block of text for tokenization
                conversation = "\n".join(chunk['messages'])
                
                # Tokenize the conversation (returns BatchEncoding object)
                tokens = self.tokenizer(
                    conversation,
                    truncation=True,
                    padding="max_length",
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
                
                
                
                # Convert BatchEncoding (tensor) to Python lists for JSON serialization
                tokens_dict = {
                    key: val.cpu().tolist()[0]  # Convert tensors to lists, this is 1 item in a list to extract to avoid mismatch in fine tuning tensors
                    for key, val in tokens.items()
                }
                
                tokens_dict["labels"] = tokens_dict["input_ids"].copy()
                
                tokenized_data.append(tokens_dict)
                data.append(chunk['messages'])

            # Save tokenized data to disk or use it directly
            with open(tokenized_output_file, 'a') as f:
                for entry in tokenized_data:
                    json.dump(entry, f)
                    f.write('\n')
                    
            # Save data to disk
            with open(output_file, 'a') as f:
                for entry in data:
                    json.dump(entry, f)
                    f.write(f'\n')
                    
                    
        buffer = []  # To accumulate and batch-process sliding windows
        batch_size = 100  # Tokenize every 100 chunks to avoid memory overflow
        
        for _, data in tqdm(enumerate(dataset)):
            buffer.append(data)
            
            # Once buffer reaches batch size, apply sliding window
            if len(buffer) >= batch_size:
                all_chunks = []
                
                # Process each conversation/message batch in sliding windows
                for chat in buffer:
                    chunks = sliding_window([chat])
                    all_chunks.extend(chunks)
                
                # Tokenize and save in batches
                tokenize_and_save(all_chunks)
                buffer = []  # Clear buffer after processing

        # Handle remaining buffer
        if buffer:
            all_chunks = []
            for chat in buffer:
                chunks = sliding_window([chat])
                all_chunks.extend(chunks)
            tokenize_and_save(all_chunks)
            
    # Load the tokenized data from JSONL file
    def load_tokenized_data(self):
        file_path = f'{self.config["data"]["folder"]}/{self.config["data"]["tokenized"]}'
        data = []
        
        # Read each line (which is a tokenized entry)
        with open(file_path, 'r') as f:
            for line in f:
                # Convert JSON string back to dictionary
                data.append(json.loads(line))
        
        # Create a Hugging Face Dataset from the list of dictionaries
        return Dataset.from_list(data)
    
            
    def setup_model(self):
        
        # Load the model with quantization and auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",  # Automatically uses GPU, with CPU fallback if necessary
            use_cache=False,  # Disable cache for compatibility with gradient checkpointing
        )
        
        # model = model.to_empty(device="cuda")
        
        # model.gradient_checkpointing_enable()
        # model.gradient_checkpointing_disable()
        
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
        if self.config["model"]["use_quantization"]:
            model = prepare_model_for_kbit_training(model)
        
        # Wrap the base model with LoRA adapters
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Confirm trainable parameters
        
        self.model = model
        
        

    def train(self):
        
        tokenized_dataset = self.load_tokenized_data()
        
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.05)

        # Access train and eval datasets
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        # Verify sizes
        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        
        # Load evaluation metric (perplexity or accuracy)
        metric = load("perplexity")

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            
            # Ensure logits are converted to probabilities
            if isinstance(logits, tuple):
                logits = logits[0]  # For models returning tuples

            perplexity = metric.compute(predictions=predictions, references=labels)
            perplexity["perplexity"] = np.exp(perplexity["loss"])  # Convert loss to perplexity
            return perplexity
        
        lr = self.config["training"]["learning_rate"]
        epochs = self.config["training"]["num_epochs"]
        weight_decay = self.config["training"]["weight_decay"]
        
        logging_strategy = self.config["training"]["logging_strategy"]
        save_strategy = self.config["training"]["save_strategy"]
        eval_strategy = self.config["training"]["eval_strategy"]
        
        logging_steps = self.config["training"]["logging_steps"]
        save_steps = self.config["training"]["save_steps"]
        eval_steps = self.config["training"]["eval_steps"]
        
        report_to = []
        if self.config["logging"]["use_wandb"]:
            report_to.append("wandb")
            
        if self.config["logging"]["use_tensorboard"]:
            report_to.append("tensorboard")
        
        training_args = TrainingArguments(
            output_dir=f"./fine-tuned-group-chat/{self.run_name}",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,  # Smaller evaluation batch size
            
            num_train_epochs=epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            fp16=True,
            
            logging_strategy=logging_strategy,
            # eval_strategy=eval_strategy,  # Ensure evaluation happens at the same interval
            save_strategy=save_strategy,  # Match the saving interval with evaluation
            
            logging_steps=logging_steps,  # Log progress every 50 steps
            # eval_steps=eval_steps,
            save_steps=save_steps,
            
            logging_dir=f"./logs/{self.run_name}",  # Path for TensorBoard logs
            # log_level="info",  # Log information at INFO level
            report_to=report_to,  # Report metrics to TensorBoard, launch with tensorboard --logdir ./logs
            run_name=self.run_name,
            
            disable_tqdm=False,
            # load_best_model_at_end=True,  # Required for EarlyStoppingCallback
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                # EarlyStoppingCallback(early_stopping_patience=3),
                ClearCacheCallback()
                ]
        )
        
        torch.cuda.empty_cache()
        self.model.train()
        try:
            trainer.train()
        except:
            print("Ending training")
        
        
        
if __name__ == "__main__":
    model = FineTunedChat()
    
    print("Processing data")
    model.process_data()
    
    print("Setting up model")
    model.setup_model()
    
    print("Training")
    model.train()
    
    wandb.finish()