#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spincore/Spiron LLM Integration Demo
====================================

This script demonstrates how to use a Spincore neuron and/or a
Spiron network to bias the output of a large language model (LLM).
It leverages the ``align_with_llm`` method introduced in
``spincore_neuron.py`` to integrate dynamically computed control
vectors into the generation process.  The example uses the
``distilgpt2`` model for brevity, but any causal language model
compatible with the HuggingFace ``transformers`` library should work.

Key features:

* **Spiron network** – A web of spinning points that flips binary
  states at regular intervals.  The aggregated states form a control
  vector that is projected into the model's logits at generation
  time.
* **Robust sampling** – Generates text using nucleus sampling and
  temperature scaling with additional penalties to discourage
  repetition.

To run this demo, ensure ``torch`` and ``transformers`` are installed.

Usage:

    python spincore_demo.py

Press Ctrl+C or type ``exit`` to quit the interactive loop.
"""

import sys
import time
import socket
from typing import Optional

import numpy as np

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: This demo requires PyTorch and transformers. Please install them with:\n"
          "    pip install torch transformers")
    sys.exit(1)

from spincore_neuron import SpincoreNeuron
from spiron_network import SpironNetwork


def check_internet_connection() -> bool:
    """Check if there is an active internet connection."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except OSError:
        return False


def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    """Load a HuggingFace model and tokenizer with optional caching.

    If the model files exist in ``cache_dir``, they are loaded from
    there; otherwise they are downloaded from HuggingFace (internet
    connection required).

    Args:
        model_name: Pretrained model identifier (e.g. ``"distilgpt2"``).
        cache_dir: Optional directory to cache model files.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    # Attempt to load model and tokenizer from cache_dir
    if cache_dir:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            model = AutoModelForCausalLM.from_pretrained(cache_dir)
            return model, tokenizer
        except Exception:
            pass
    # Fallback: load from HuggingFace
    if not check_internet_connection():
        raise RuntimeError("No internet connection and model not found in cache_dir.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def main() -> None:
    model_name = "distilgpt2"
    cache_dir = f"./{model_name}_cache"
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, cache_dir=cache_dir)
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        sys.exit(1)

    # Ensure pad token exists (needed for generation when using logits processors)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate a Spiron network with an even number of nodes equal to the
    # control dimension used by the Spincore neuron.  The number of nodes
    # determines the dimensionality of the control vector.  Here we use 64
    # nodes to balance expressiveness and computational cost.
    num_nodes = 64
    spiron = SpironNetwork(num_nodes=num_nodes, radius=1.0, interval=0.025, weighted=True)

    # Create a Spincore neuron only to satisfy the API; its output_dim
    # should match the number of nodes if no Spiron network is provided.
    # Since we're using a Spiron network, the neuron will not be used to
    # produce control vectors, but ``align_with_llm`` still belongs to
    # SpincoreNeuron.  The input_dim is chosen to accommodate flattening
    # of a few token embeddings (e.g. 5 tokens).  Adjust as needed.
    hidden_size = model.config.hidden_size
    input_dim = hidden_size * 5
    neuron = SpincoreNeuron(input_dim=input_dim, output_dim=num_nodes)

    # Obtain a generation function that biases logits with the Spiron output
    generate = neuron.align_with_llm(
        model,
        tokenizer=tokenizer,
        freeze_spincore=True,  # freeze orientation updates since we use Spiron
        mode="logits",
        spiron_network=spiron,
        scale=0.03,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    # Interactive loop
    print("Spiron‑enhanced LLM demo.  Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            # Generate with sampling parameters for diversity
            output_ids = generate(
                user_input,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.85,
            )
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\nSpiron: {response}")
        except Exception as exc:
            print(f"Error during generation: {exc}")


if __name__ == "__main__":
    main()