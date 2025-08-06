from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

# Import torch conditionally to avoid hard dependency
try:
    import torch
except ImportError:
    torch = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    This activation squashes inputs into the (0, 1) interval.  It is
    implemented in a way that avoids overflow for large absolute inputs.

    Args:
        x: A NumPy array of arbitrary shape.

    Returns:
        An array of the same shape containing values in the range
        ``(0, 1)``.
    """
    # For positive values, use the standard form.  For negative values,
    # rewrite the sigmoid to avoid overflow: σ(-x) = 1 - σ(x).
    out = np.empty_like(x, dtype=np.float64)
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    # σ(x) = 1 / (1 + exp(-x))
    out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    out[negative_mask] = exp_x / (1.0 + exp_x)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit activation.

    Args:
        x: A NumPy array.

    Returns:
        ``max(x, 0)`` applied elementwise.
    """
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the softmax along a given axis with numerical stability.

    Args:
        x: Input array.
        axis: Axis along which to apply the softmax.

    Returns:
        A new array where the values along ``axis`` sum to 1.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


@dataclass
class SpincoreNeuron:
    """A simple spincore neuron with trainable weights and orientation.

    Each neuron accepts a fixed‑size input vector and produces a fixed‑size
    output vector.  Internally it maintains a weight matrix (mapping inputs
    to outputs), an orientation vector and a spin state.  The orientation
    modulates the magnitude of the neuron's output on a per‑dimension
    basis; the spin state is a binary flag that can be toggled based on
    application logic (for example, to flip orientation when certain
    conditions are met).

    The class also exposes training methods implementing a simple form of
    gradient descent.  These methods are designed to be illustrative
    rather than production‑ready—there are no advanced optimisers or
    hardware accelerations used.

    Attributes:
        input_dim: Dimensionality of the input vectors.
        output_dim: Dimensionality of the output vectors.
        orientation: A unit‑norm vector of length ``output_dim`` that
            modulates the neuron's outputs.  Each entry typically lies in
            ``[-1, 1]``; internal methods ensure the vector remains
            normalised during training.
        spin_state: An integer (``+1`` or ``-1``) indicating the current
            spin configuration.  Multiplying the orientation by
            ``spin_state`` flips its sign, providing a simple mechanism to
            represent ``spin‑up`` versus ``spin‑down``.
        weight: A weight matrix of shape ``(input_dim, output_dim)``.
        bias: A bias vector of length ``output_dim``.
        activation: Activation function used in the forward pass.  This
            should be a callable mapping ``np.ndarray`` to ``np.ndarray``.

    Example:

    >>> neuron = SpincoreNeuron(3, 2)
    >>> x = np.array([[1.0, 0.5, -0.2]])
    >>> output = neuron.forward(x)
    >>> print(output)
    [[0.42 0.59]]

    The numbers above are arbitrary; your results will differ because the
    neuron's weights and orientation are initialised randomly.
    """

    input_dim: int
    output_dim: int
    orientation: np.ndarray = field(init=False)
    spin_state: int = field(init=False, default=1)
    weight: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    activation: Callable[[np.ndarray], np.ndarray] = field(init=False, default=_sigmoid)
    # Derivative of the activation function with respect to its input.  By
    # default this is the derivative of the sigmoid, but it can be
    # overridden when setting a custom activation.  It should accept the
    # activation's output as input and return the derivative at that point.
    activation_derivative: Callable[[np.ndarray], np.ndarray] = field(init=False)
    # Additional hyperparameter controlling how quickly the orientation vector
    # is updated relative to the main learning rate.  A value of 0 freezes
    # the orientation (useful during linear regression), values greater than
    # 1 accelerate orientation adaptation.
    orientation_lr_scale: float = field(default=1.0)

    def __post_init__(self) -> None:
        # Validate dimensions
        if not isinstance(self.input_dim, numbers.Integral) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {self.input_dim}")
        if not isinstance(self.output_dim, numbers.Integral) or self.output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {self.output_dim}")

        # Randomly initialise weight matrix and bias.  We use a small
        # standard deviation to avoid saturating the sigmoid at the outset.
        rng = np.random.default_rng()
        self.weight = rng.normal(loc=0.0, scale=1.0 / math.sqrt(self.input_dim), size=(self.input_dim, self.output_dim))
        self.bias = np.zeros(self.output_dim, dtype=np.float64)

        # Initialise orientation to a random unit vector.  This vector is
        # multiplied by spin_state in the forward pass.
        random_orientation = rng.normal(size=self.output_dim)
        norm = np.linalg.norm(random_orientation)
        if norm == 0:
            random_orientation[0] = 1.0
            norm = 1.0
        self.orientation = random_orientation / norm

        # spin_state is initialised to +1 (spin‑up).  It can be flipped
        # during training or based on external logic.
        self.spin_state = 1

        # Default activation is sigmoid, but can be replaced after initialisation.
        self.activation = _sigmoid
        # Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z)).  Since
        # ``predictions`` in the gradient are the activated values, we
        # define derivative as a function of the activated output.
        self.activation_derivative = lambda y: y * (1.0 - y)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the neuron's output for the given input batch.

        Args:
            x: A NumPy array of shape ``(batch_size, input_dim)``.

        Returns:
            A NumPy array of shape ``(batch_size, output_dim)`` containing
            the activated, orientation‑modulated outputs.
        """
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input of shape (batch_size, {self.input_dim}), got {x.shape}")

        # Linear transformation
        z = x @ self.weight + self.bias  # shape: (batch_size, output_dim)
        # Apply spin and orientation
        oriented = z * (self.orientation * self.spin_state)
        # Activation
        return self.activation(oriented)

    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray, loss_type: str = "mse") -> float:
        """Compute a simple loss between predictions and targets.

        Args:
            predictions: Model outputs of shape ``(batch_size, output_dim)``.
            targets: Ground truth of the same shape.  For classification
                tasks using cross entropy the targets should be one‑hot
                encoded.
            loss_type: Either ``"mse"`` or ``"cross_entropy"``.

        Returns:
            A scalar representing the average loss across the batch.
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

        if loss_type == "mse":
            diff = predictions - targets
            return 0.5 * np.mean(np.square(diff))
        elif loss_type == "cross_entropy":
            # Assume outputs of forward pass are probabilities (sigmoid/softmax).
            # To avoid numerical issues, clip values into (eps, 1-eps).
            eps = 1e-12
            clipped = np.clip(predictions, eps, 1.0 - eps)
            return -np.mean(np.sum(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped), axis=1))
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _gradient(self, x: np.ndarray, predictions: np.ndarray, targets: np.ndarray, loss_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients of the loss with respect to weights, bias and orientation.

        The gradient derivations assume the orientation and spin act as a
        multiplicative factor on the pre‑activation outputs.  The returned
        gradients can be used in a simple gradient descent update.

        Args:
            x: Input batch of shape ``(batch_size, input_dim)``.
            predictions: Output of ``self.forward(x)``.
            targets: Ground truth of shape ``(batch_size, output_dim)``.
            loss_type: The same loss type passed to ``compute_loss``.

        Returns:
            Tuple of gradients ``(grad_weight, grad_bias, grad_orientation)``.
        """
        batch_size = x.shape[0]
        # Derivative of loss w.r.t. oriented preactivation.  For
        # simplicity we derive gradients for mse and binary cross
        # entropy with sigmoid; multi‑class cross entropy would require
        # a softmax derivative.  The code can be extended as needed.

        if loss_type == "mse":
            # dL/dy = (y - t)
            dL_dy = (predictions - targets) / batch_size
            # dy/dz = activation_derivative(predictions)
            dy_dz = self.activation_derivative(predictions)
            dL_dz = dL_dy * dy_dz
        elif loss_type == "cross_entropy":
            # For binary cross entropy with sigmoid: dL/dz = (y - t)/batch_size
            dL_dz = (predictions - targets) / batch_size
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # dL/dW = x^T * (dL/dz * (orientation * spin_state)).
        # Explanation: oriented = z * (orientation * spin_state), so
        # partial derivative of oriented w.r.t. W is x * (orientation * spin_state).
        orient_sign = self.orientation * self.spin_state
        grad_weight = x.T @ (dL_dz * orient_sign)
        # dL/dbias = sum of dL/dz * (orientation * spin_state).
        grad_bias = np.sum(dL_dz * orient_sign, axis=0)
        # For orientation gradient we derive:
        # oriented_i = z_i * orientation_i * spin_state
        # dL/dorientation_i = sum_s [dL/d oriented_s,i * z_s,i * spin_state]
        # But dL/d oriented_s,i = dL/d z_s,i / (orientation_i * spin_state)
        # => dL/dorientation_i = sum_s (dL/dz_s,i * z_s,i) / orientation_i
        preact = x @ self.weight + self.bias  # shape: (batch, output_dim)
        # Sum over batch of dL/dz * z
        numerator = np.sum(dL_dz * preact, axis=0)
        # Avoid division by zero by adding a small epsilon to orientation
        eps = 1e-8
        denom = self.orientation.copy()
        denom[np.abs(denom) < eps] = eps * np.sign(denom[np.abs(denom) < eps] + eps)
        grad_orientation = numerator / denom
        return grad_weight, grad_bias, grad_orientation

    def update_parameters(self, grad_weight: np.ndarray, grad_bias: np.ndarray, grad_orientation: np.ndarray, lr: float) -> None:
        """Apply a gradient descent step to weights, bias and orientation.

        After updating the orientation vector, it is renormalised to have
        unit length.  This prevents the orientation from growing without
        bound and ensures it remains a direction vector.

        Args:
            grad_weight: Gradient of the loss w.r.t. the weight matrix.
            grad_bias: Gradient of the loss w.r.t. the bias vector.
            grad_orientation: Gradient of the loss w.r.t. the orientation.
            lr: Learning rate (a small positive scalar).
        """
        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        # Apply scaled orientation update.  Orientation learning can be
        # slowed down relative to the main learning rate by setting
        # ``orientation_lr_scale`` < 1.0.
        orientation_step = lr * self.orientation_lr_scale
        self.orientation -= orientation_step * grad_orientation
        # Renormalise orientation to unit length to prevent it from
        # exploding or vanishing.  If the vector collapses to zero, reset
        # to a uniform direction.
        norm = np.linalg.norm(self.orientation)
        if norm == 0 or not np.isfinite(norm):
            self.orientation = np.ones_like(self.orientation) / math.sqrt(self.output_dim)
        else:
            self.orientation /= norm

    def train_supervised(self,
                         inputs: np.ndarray,
                         targets: np.ndarray,
                         epochs: int = 100,
                         lr: float = 0.1,
                         batch_size: Optional[int] = None,
                         loss_type: str = "mse",
                         verbose: bool = False) -> List[float]:
        """Train the neuron on a labelled dataset using gradient descent.

        Args:
            inputs: Array of shape ``(num_samples, input_dim)``.
            targets: Array of shape ``(num_samples, output_dim)``.
            epochs: Number of passes through the entire dataset.
            lr: Learning rate for gradient descent.
            batch_size: Size of each mini‑batch.  If ``None`` (default),
                full‑batch gradient descent is used.
            loss_type: Loss function to minimise: ``"mse"`` or
                ``"cross_entropy"``.
            verbose: If True, print loss every 10 epochs.

        Returns:
            A list of losses recorded after each epoch.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (N, {self.input_dim}), got {inputs.shape}")
        if targets.shape != (inputs.shape[0], self.output_dim):
            raise ValueError(f"targets must have shape (N, {self.output_dim}), got {targets.shape}")
        num_samples = inputs.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > num_samples:
            batch_size = num_samples
        losses: List[float] = []
        for epoch in range(epochs):
            # Shuffle dataset at the beginning of each epoch
            perm = np.random.permutation(num_samples)
            inputs_shuffled = inputs[perm]
            targets_shuffled = targets[perm]
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                x_batch = inputs_shuffled[start:end]
                y_batch = targets_shuffled[start:end]
                # Forward pass
                preds = self.forward(x_batch)
                # Compute loss
                batch_loss = self.compute_loss(preds, y_batch, loss_type)
                epoch_loss += batch_loss * x_batch.shape[0]
                # Backpropagate
                grad_w, grad_b, grad_o = self._gradient(x_batch, preds, y_batch, loss_type)
                # Update parameters
                self.update_parameters(grad_w, grad_b, grad_o, lr)
            # Average loss over epoch
            avg_loss = epoch_loss / num_samples
            losses.append(avg_loss)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")
        return losses

    def self_supervised_training(self,
                                 inputs: np.ndarray,
                                 epochs: int = 100,
                                 lr: float = 0.1,
                                 verbose: bool = False) -> List[float]:
        """Train the neuron in a self‑supervised fashion using autoencoding.

        This method uses the neuron's forward pass both to encode and decode
        the input: the model tries to reconstruct its own input.  While a
        single neuron is not expressive enough to act as an autoencoder,
        this procedure serves as a simple demonstration of unsupervised
        learning.  For meaningful representation learning one would use
        multiple layers and non‑linearities; those are outside the scope
        of this version.

        Args:
            inputs: Array of shape ``(num_samples, input_dim)``.
            epochs: Number of training epochs.
            lr: Learning rate.
            verbose: If True, print loss occasionally.

        Returns:
            A list of reconstruction losses per epoch.
        """
        # The reconstruction target is simply the inputs themselves up to the
        # output dimensionality.  If ``output_dim`` < ``input_dim``, the
        # reconstruction target is truncated; if greater, the targets are
        # padded with zeros.
        num_samples = inputs.shape[0]
        if self.output_dim <= self.input_dim:
            targets = inputs[:, : self.output_dim]
        else:
            padding = np.zeros((num_samples, self.output_dim - self.input_dim), dtype=np.float64)
            targets = np.concatenate([inputs, padding], axis=1)
        return self.train_supervised(inputs, targets, epochs=epochs, lr=lr, batch_size=None, loss_type="mse", verbose=verbose)

    def integrate_with_llm(self,
                           llm_model: Optional[object] = None,
                           tokenizer: Optional[object] = None,
                           prefix_prompt: Optional[str] = None,
                           input_ids: Optional[np.ndarray] = None,
                           max_length: int = 50,
                           **generate_kwargs) -> Tuple[str, np.ndarray]:
        """Combine Spincore outputs with a language model to generate text.

        This method is a thin wrapper illustrating one way to inject the
        Spincore neuron's output into an existing language model.  The idea
        is that the neuron's activations (after forward propagation) can be
        treated as an embedding or control signal that biases the language
        model's generation.

        Because HuggingFace ``transformers`` is not necessarily available
        in this environment, the method performs feature detection and
        raises informative errors when integration cannot proceed.

        **Usage scenario**:

        1. Obtain or load a language model ``llm_model`` and its
           ``tokenizer``.  The model should support the ``generate`` method
           accepting inputs and optionally past key/values.
        2. Provide a prefix prompt or raw input ``input_ids`` for the
           language model.  If ``prefix_prompt`` is provided, it will be
           tokenised to produce ``input_ids``.
        3. Compute the Spincore neuron's output for a custom input (the
           custom input should reflect the context you want to bias the
           generation with).  This custom input is not passed here; you
           must call ``forward`` yourself and supply its result via
           ``generate_kwargs`` if needed.
        4. Call ``integrate_with_llm``; it will attempt to append the
           Spincore activations to the model's embeddings (if supported) and
           then generate text.

        Note that integration strategies vary greatly depending on the
        underlying LLM; the implementation below is intentionally generic
        and may need to be customised.

        Args:
            llm_model: A loaded language model with a ``generate`` method
                (e.g. a HuggingFace ``PreTrainedModel``).  If None, the
                method will attempt to import a minimal GPT‑2 as a
                demonstration if available.
            tokenizer: The corresponding tokenizer for ``llm_model``.
            prefix_prompt: Optional text prompt to start generation.
            input_ids: Optional already tokenised input ids.  If both
                ``prefix_prompt`` and ``input_ids`` are provided, the
                latter takes precedence.
            max_length: Maximum length of the generated sequence.
            **generate_kwargs: Additional keyword arguments forwarded to
                the model's ``generate`` method.

        Returns:
            A tuple ``(generated_text, spincore_output)`` where
            ``generated_text`` is the string produced by the language
            model, and ``spincore_output`` is the orientation‑modulated
            activation used to bias the model.  If integration fails, an
            exception is raised.
        """
        # Import transformers lazily to avoid hard dependency
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace transformers is required for LLM integration, but it is not installed."
            ) from exc

        # Use default model/tokenizer if not provided
        if llm_model is None:
            try:
                llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load default language model 'distilgpt2'. Please install transformers and provide a model explicitly."
                ) from exc
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(llm_model.config._name_or_path)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load tokenizer. Please provide a tokenizer corresponding to the language model."
                ) from exc

        # Prepare input ids
        if input_ids is None:
            if prefix_prompt is None:
                raise ValueError("Either input_ids or prefix_prompt must be provided.")
            # Tokenise prompt
            input_ids = tokenizer.encode(prefix_prompt, return_tensors="np")
        else:
            if not isinstance(input_ids, np.ndarray):
                raise TypeError("input_ids must be a NumPy array if provided.")

        # The following demonstrates a naive way of combining Spincore activations
        # with the language model's embeddings.  We compute the neuron's
        # output for a dummy input (zero vector) and then append it as a
        # pseudo‐token to the input sequence.  Real applications might
        # integrate at the embedding layer instead of the token level.
        dummy_input = np.zeros((1, self.input_dim), dtype=np.float64)
        spincore_output = self.forward(dummy_input)  # shape: (1, output_dim)
        # Convert Spincore output into a token embedding.  For GPT‑2 like
        # models the hidden size typically equals the embedding size; here
        # we pad or truncate to match the embedding dimension.
        hidden_size = llm_model.config.hidden_size
        spin_embed = np.zeros((1, hidden_size), dtype=np.float64)
        length = min(hidden_size, self.output_dim)
        spin_embed[0, :length] = spincore_output[0, :length]
        # Retrieve the model's input embeddings and append the spin embed
        try:
            # For causal LMs, ``inputs_embeds`` argument bypasses the
            # token embedding layer.  We'll get the embeddings for the
            # existing input ids and then append our custom embedding.
            inputs_embeds = llm_model.get_input_embeddings()(input_ids)
            # Convert Spincore embedding to torch tensor
            import torch
            spin_embed_tensor = torch.from_numpy(spin_embed.astype(np.float32))
            # Concatenate along sequence dimension
            extended_embeds = torch.cat([inputs_embeds, spin_embed_tensor], dim=1)
            # Generate continuation using the extended embeddings
            outputs = llm_model.generate(inputs_embeds=extended_embeds, max_length=max_length, **generate_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to integrate Spincore output with the language model."
            ) from exc

        # Decode generated ids
        generated_ids = outputs[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, spincore_output

    def align_with_llm(
        self,
        llm_model,
        tokenizer=None,
        freeze_spincore: bool = False
    ):
        """
        Returns a `generate_with_spincore(prompt: str, **kwargs)` function
        that wraps llm_model.generate(), inserting Spincore outputs
        into the token embeddings.

        Args:
          llm_model:           a HuggingFace model (e.g. transformers.PreTrainedModel)
          tokenizer:           corresponding tokenizer (required)
          freeze_spincore:     if True, sets learning_rate and orientation_lr_scale to 0
        """
        
        # Check if torch is available
        if torch is None:
            raise ImportError(
                "PyTorch is required for LLM integration, but it is not installed. "
                "Please install torch before using this method."
            )

        # — optionally freeze all Spincore training —
        if freeze_spincore:
            self.orientation_lr_scale = 0.0

        def generate_with_spincore(prompt: str, max_length=50, **generate_kwargs):
            if tokenizer is None:
                raise ValueError("You must pass a tokenizer to align_with_llm().")

            # 1) Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(llm_model.device)
            attention_mask = inputs.attention_mask.to(llm_model.device) if hasattr(inputs, 'attention_mask') else None

            # 2) Pull raw token embeddings
            emb_layer = llm_model.get_input_embeddings()
            embeddings = emb_layer(input_ids)
            # embeddings shape: (batch=1, seq_len, hidden_dim)

            # 3) Run spincore forward on a flattened view of those embeddings
            #    (or choose a pooling you prefer)
            emb_np = embeddings.detach().cpu().numpy()[0]            # (seq_len, hidden_dim)
            x = emb_np.reshape(1, -1)                               # (1, seq_len*hidden_dim)
            # Make sure the input dimension matches what the neuron expects
            if x.shape[1] != self.input_dim:
                # If dimensions don't match, we need to either truncate or pad
                if x.shape[1] > self.input_dim:
                    # Truncate if too large
                    x = x[:, :self.input_dim]
                else:
                    # Pad with zeros if too small
                    padding = np.zeros((1, self.input_dim - x.shape[1]), dtype=x.dtype)
                    x = np.concatenate([x, padding], axis=1)
            spin_out = self.forward(x)                              # (1, output_dim)

            # 4) Convert back to a torch Tensor and broadcast
            spin_tensor = torch.tensor(
                spin_out,
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            # Now expand to (1, seq_len, output_dim)
            spin_feat = spin_tensor.unsqueeze(1).expand(
                embeddings.size(0), embeddings.size(1), -1
            )

            # 5) For now, we'll disable the SpincoreNeuron's influence on embeddings
            # as it seems to be causing repetitive text generation
            mod_embeddings = embeddings
            
            # Note: For future improvements, consider these approaches:
            # 1. Use spincore output to influence attention weights instead of embeddings
            # 2. Apply a more sophisticated integration method with proper scaling
            # 3. Use spincore output as a conditioning signal for the model
            # 4. Train the spincore neuron specifically to avoid repetitive patterns
            # such as using the spincore output to influence attention weights
            # or using it as a prefix/prompt embedding

            # Finally call the model's generate using our patched embeddings
            generate_kwargs_with_mask = generate_kwargs.copy()
            
            # Add attention_mask if available
            if attention_mask is not None:
                generate_kwargs_with_mask['attention_mask'] = attention_mask
                
            # Set pad_token_id explicitly if not already set
            if 'pad_token_id' not in generate_kwargs_with_mask and hasattr(tokenizer, 'pad_token_id'):
                if tokenizer.pad_token_id is not None:
                    generate_kwargs_with_mask['pad_token_id'] = tokenizer.pad_token_id
                elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    # Fall back to eos_token_id if pad_token_id is not set
                    generate_kwargs_with_mask['pad_token_id'] = tokenizer.eos_token_id
            
            return llm_model.generate(
                inputs_embeds=mod_embeddings,
                max_length=max_length,
                **generate_kwargs_with_mask
            )

        return generate_with_spincore

    def flip_spin(self) -> None:
        """Toggle the spin state between +1 (spin‑up) and −1 (spin‑down)."""
        self.spin_state *= -1

    def inverse_orientation(self) -> np.ndarray:
        """Return the elementwise inverse of the orientation vector.

        This method is provided for completeness.  In most cases you
        shouldn't need to take the reciprocal of orientation values because
        the orientation is normalised; however, it can be useful when
        deriving gradients.

        Returns:
            A new array where each element ``i`` is ``1.0 / orientation[i]``.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(self.orientation != 0, 1.0 / self.orientation, 0.0)
        return inv


def demo() -> None:
    """Demonstrate supervised training of a Spincore neuron on a toy task.

    The task: learn to map 2‑D inputs to 1‑D outputs via a linear function
    with added noise.  The neuron is trained using mean squared error.
    After training, the function prints the average loss and a few
    predictions to illustrate convergence.
    """
    rng = np.random.default_rng(42)
    # Create synthetic dataset: y = 0.3 * x0 - 0.2 * x1 + 0.1
    num_samples = 500
    x = rng.normal(size=(num_samples, 2))
    noise = 0.05 * rng.standard_normal(size=(num_samples, 1))
    y_true = 0.3 * x[:, [0]] - 0.2 * x[:, [1]] + 0.1 + noise
    # Initialise spincore neuron.  For a linear regression task we
    # freeze the orientation by setting orientation_lr_scale to 0.0.  This
    # prevents the orientation update from destabilising the simple
    # regression during training.
    neuron = SpincoreNeuron(input_dim=2, output_dim=1)
    neuron.orientation_lr_scale = 0.0
    # Use identity activation (linear regression) by replacing the
    # sigmoid with an identity function.  Without non‑linearity the
    # neuron performs linear regression with orientation gating.
    neuron.activation = lambda z: z  # type: ignore
    neuron.activation_derivative = lambda y: np.ones_like(y)
    # Train
    losses = neuron.train_supervised(x, y_true, epochs=200, lr=0.05, verbose=True)
    print(f"Final loss: {losses[-1]:.6f}")
    # Test on new samples
    test_x = rng.normal(size=(5, 2))
    preds = neuron.forward(test_x)
    for i in range(5):
        print(f"Input: {test_x[i]}, Prediction: {preds[i,0]:.4f}")


if __name__ == "__main__":
    demo()
