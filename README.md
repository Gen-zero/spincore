# spincore
This module defines a first–pass implementation of a *Spincore* neuron and
associated training routines.  The aim of the Spincore concept is to form
the backbone of an aligned general intelligence (AGI) architecture without
making assumptions about any particular downstream model.  In practice
Spincore neurons provide a simple, differentiable substrate that can be
trained on data and optionally interfaced with existing large language
models (LLMs).

## **Design goals**

1. **Minimal dependencies** – the code relies only on the Python standard
   library and NumPy.  Optional integrations with external frameworks such
   as PyTorch or HuggingFace are provided behind feature‐checks.  If those
   libraries are unavailable, the core of Spincore continues to work.
2. **Transparency** – weights, orientation vectors and spin state are all
   explicit and can be inspected or saved for later analysis.  There are
   no hidden globals or side effects.
3. **Flexibility** – the neuron can be trained in supervised, self‑supervised
   or reinforcement settings.  Additional layers of training logic can be
   built on top of the provided primitives without modifying the core
   implementation.
4. **Compatibility** – where possible the class exposes hooks for feeding
   Spincore activations into other neural architectures.  The
   ``integrate_with_llm`` method illustrates how to pass Spincore outputs
   as prefixes or prompts to a language model if such a model is
   available.

This file is deliberately self‑contained so that it can be dropped into
other projects without dragging along a large dependency tree.  It is not
meant to be the final implementation; rather, it lays the groundwork for
future research into aligned and interpretable neural components.

## **Using SpincoreNeuron with Language Models**

The SpincoreNeuron class now includes an `align_with_llm` method that provides a more direct integration with Hugging Face transformer models. This method returns a function that modulates the LLM's token embeddings with the neuron's output.

### Requirements

To use the LLM integration features, you'll need:

```
pip install torch transformers
```

### Basic Usage

Here's a simple example of how to use the `align_with_llm` method:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from spincore_neuron import SpincoreNeuron

# Load model and tokenizer
llm = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")

# Create and configure your neuron
neuron = SpincoreNeuron(input_dim=768*5, output_dim=64)  # Adjust dimensions as needed

# Get the generation function
generate = neuron.align_with_llm(llm, tokenizer=tok, freeze_spincore=False)

# Generate text with spincore modulation
out = generate("Once upon a time,")
print(tok.decode(out[0], skip_special_tokens=True))
```

### Interactive Demo

A demo script `spincore_demo.py` is included to demonstrate interactive conversation with a SpincoreNeuron-enhanced language model:

```
python spincore_demo.py
```

This will start an interactive session where you can converse with the model. The SpincoreNeuron will modulate the language model's embeddings based on your inputs, potentially guiding or biasing the generation process.

### How It Works

The `align_with_llm` method:

1. Takes the token embeddings from the language model
2. Runs them through the SpincoreNeuron's forward pass
3. Concatenates the SpincoreNeuron's output with the original embeddings
4. Feeds these modified embeddings back to the language model for generation

This allows the SpincoreNeuron to influence the language model's generation process in a way that reflects the neuron's learned patterns and orientation.
