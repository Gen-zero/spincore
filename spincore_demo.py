#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpincoreNeuron LLM Integration Demo

This script demonstrates how to use the SpincoreNeuron class with a language model
for interactive conversation. It loads a pre-trained language model and tokenizer,
initializes a SpincoreNeuron, and sets up an interactive loop for conversation.

Usage:
    python spincore_demo.py

Requirements:
    - torch
    - transformers
    - numpy
"""

import sys
import os
import socket
import numpy as np
from spincore_neuron import SpincoreNeuron

# Import required libraries for LLM integration
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    HAS_REQUIREMENTS = True
except ImportError:
    print("Please install required packages with: pip install torch transformers huggingface_hub")
    HAS_REQUIREMENTS = False

def check_internet_connection():
    """
    Check if there is an active internet connection by attempting to connect to HuggingFace.
    """
    try:
        # Try to connect to huggingface.co
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except OSError:
        return False

def main():
    if not HAS_REQUIREMENTS:
        print("This demo requires PyTorch and transformers. Please install with:")
        print("pip install torch transformers")
        sys.exit(1)
    
    # Use a smaller model (distilgpt2) for faster download and loading
    model_name = "distilgpt2"  # Smaller than gpt2
    cache_dir = f"./{model_name}_model_cache"
    
    print(f"Starting Spincore demo with {model_name.upper()} model")
    print(f"Model files will be cached in: {cache_dir}")
    print("-" * 50)
    
    # Check internet connection if we need to download the model
    if not os.path.exists(cache_dir) or not any([
        os.path.exists(os.path.join(cache_dir, "pytorch_model.bin")),
        os.path.exists(os.path.join(cache_dir, "model.safetensors"))
    ]):
        if not check_internet_connection():
            print("Warning: No internet connection detected. Cannot download the model.")
            print("Please ensure you have an active internet connection and try again.")
            sys.exit(1)
    
    print(f"Loading {model_name} model and tokenizer...")
    # Load tokenizer from local directory if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"./{model_name}")
        print("✓ Tokenizer loaded from local directory")
    except Exception as e:
        print(f"Could not load tokenizer from local directory: {e}")
        print("Downloading tokenizer from HuggingFace instead...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ Tokenizer downloaded successfully from HuggingFace")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
    
    # Set pad token to eos token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load model from cache or download from HuggingFace
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        except Exception as e:
            print(f"Warning: Could not create cache directory: {e}")
    
    # Check if the model cache directory exists and contains model files
    model_files_exist = os.path.exists(cache_dir) and any([
        os.path.exists(os.path.join(cache_dir, "pytorch_model.bin")),
        os.path.exists(os.path.join(cache_dir, "model.safetensors"))
    ])
    
    if model_files_exist:
        print(f"Loading model from cache directory: {cache_dir}")
        try:
            model = AutoModelForCausalLM.from_pretrained(cache_dir)
            print("✓ Model loaded successfully from cache")
        except Exception as e:
            print(f"Error loading model from cache: {e}")
            print("Downloading model from HuggingFace instead...")
            try:
                  # Use snapshot_download to download the model files directly
                  print("Using huggingface_hub to download model files...")
                  model_path = snapshot_download(
                      repo_id=model_name,
                      local_dir=cache_dir,
                      local_dir_use_symlinks=False,
                      max_workers=1,  # Limit concurrent downloads
                      tqdm_class=None  # Disable progress bar for cleaner output
                  )
                  print(f"Downloaded model files to {model_path}")
                  
                  # Now load the model from the downloaded files
                  model = AutoModelForCausalLM.from_pretrained(model_path)
                  print(f"✓ Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error downloading/loading model: {e}")
                sys.exit(1)
    else:
        print("Downloading model from HuggingFace...")
        try:
            # Use snapshot_download to download the model files directly
            print("Using huggingface_hub to download model files...")
            model_path = snapshot_download(
                repo_id=model_name,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
                max_workers=1,  # Limit concurrent downloads
                tqdm_class=None  # Disable progress bar for cleaner output
            )
            print(f"Downloaded model files to {model_path}")
            
            # Now load the model from the downloaded files
            model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error downloading/loading model: {e}")
            sys.exit(1)
    
    # Create a SpincoreNeuron
    # For GPT-2, hidden size is 768, we'll use a smaller output dimension
    # The input dimension needs to be large enough to handle the flattened token embeddings
    embedding_dim = model.config.hidden_size  # For GPT-2, this is 768
    input_dim = embedding_dim * 5  # Process 5 tokens at once when flattened
    output_dim = 64     # Smaller dimension for the spincore output
    
    print(f"Creating SpincoreNeuron with input_dim={input_dim}, output_dim={output_dim}")
    neuron = SpincoreNeuron(
        input_dim=input_dim, 
        output_dim=output_dim
    )
    # The activation is already set to sigmoid by default
    # Set learning rate for training if needed later
    
    # Get the generation function from align_with_llm
    # This function will use the SpincoreNeuron to modulate the LLM's token embeddings
    print("Setting up SpincoreNeuron-LLM integration...")
    generate = neuron.align_with_llm(
        model,                  # The language model
        tokenizer=tokenizer,    # The tokenizer
        freeze_spincore=False,  # Allow the neuron to learn during conversation
    )
    
    # These parameters will be passed to the generate function, not align_with_llm
    generation_params = {
        "max_length": 100,         # Maximum length of generated text
        "num_return_sequences": 1, # Number of different sequences to generate
        "do_sample": True,        # Enable sampling for more diverse outputs
        "top_k": 50              # Sample from top k most likely tokens
        # Note: temperature and top_p are not valid for this model configuration
    }
    
    # Interactive conversation loop
    print(f"\nSpincore-enhanced {model_name.upper()} Conversation")
    print("Type 'exit' to end the conversation")
    print("-" * 50)
    
    # We could maintain conversation history for more context
    # conversation_history = ""
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Generate response with spincore modulation
        try:
            # The generate function handles tokenization and embedding modulation
            output_ids = generate(user_input, **generation_params)
            
            # Decode the output
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\nSpincore-{model_name.upper()}: {response}")
            
            # If you want to maintain conversation history, uncomment and modify:
            # conversation_history += f"User: {user_input}\nAssistant: {response}\n"
            # Then pass conversation_history to generate() instead of user_input
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Try a different input or restart the program.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)