# Attention Module

## Overview
The attention module provides self-attention and transformer mechanisms for temporal sequence processing. It implements standard PyTorch attention layers with convenient wrappers for handling dimensional permutations required for temporal data (batch, channels, timesteps).

## Architecture & Design Patterns
- **Wrapper Pattern**: Simplifies PyTorch's MultiheadAttention and Transformer layers with automatic dimension handling
- **Temporal Processing**: Designed for time-series data with automatic permutation between (batch, channels, timesteps) and (timesteps, batch, channels)
- **Composition**: Small, focused modules that can be composed into larger architectures
- **Minimal Abstraction**: Thin wrappers around PyTorch built-ins for maximum compatibility

Key architectural decisions:
- Automatic dimension permutation for temporal data (optional via `permute` flag)
- Standard PyTorch attention mechanisms without custom implementations
- Dropout built-in for regularization
- Encoder/decoder separation following transformer architecture

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `SelfAttentionBlock` | Class | Self-attention for temporal sequences | N/A |
| `TransformerEncoderBlock` | Class | Transformer encoder for temporal data | N/A |
| `TransformerDecoderBlock` | Class | Transformer decoder for temporal data | N/A |

### Internal Components
None - all components are public API.

### Protocols/Interfaces
None - standard nn.Module interface.

## Dependencies

### Internal Dependencies
None - standalone module.

### External Dependencies
- `torch.nn`: PyTorch neural network modules (MultiheadAttention, TransformerEncoder, TransformerDecoder)

## Key Components

### Component 1: `SelfAttentionBlock`

**Purpose**: Applies multi-head self-attention to temporal sequences with optional automatic dimension permutation for convenience when working with (batch, channels, timesteps) data.

**Parameters**:
- `embed_dim: int` - Embedding dimension for attention mechanism (must match input channel dimension)
- `num_heads: int = 1` - Number of parallel attention heads (embed_dim must be divisible by num_heads)
- `permute: bool = True` - Whether to automatically permute dimensions from (batch, channels, timesteps) to (timesteps, batch, channels) and back

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, channels, timesteps) if permute=True, else (timesteps, batch, channels)
- Output: `Tensor` - Same shape as input after attention application

**Example**:
```python
from dlkit.core.models.nn.attention import SelfAttentionBlock
import torch

# Create attention block for 64-dimensional embeddings
attention = SelfAttentionBlock(embed_dim=64, num_heads=4)

# Process temporal data (batch_size=32, channels=64, timesteps=100)
x = torch.randn(32, 64, 100)
output = attention(x)  # Shape: (32, 64, 100)

# Without automatic permutation (for pre-permuted data)
attention_no_perm = SelfAttentionBlock(embed_dim=64, num_heads=4, permute=False)
x_permuted = torch.randn(100, 32, 64)  # (timesteps, batch, channels)
output = attention_no_perm(x_permuted)  # Shape: (100, 32, 64)
```

**Implementation Notes**:
- Uses PyTorch's `nn.MultiheadAttention` with 0.1 dropout
- Permutes dimensions: (batch, channels, timesteps) → (timesteps, batch, channels) for attention
- Self-attention: query, key, and value are all the same tensor
- Attention weights are not returned (only the transformed values)

---

### Component 2: `TransformerEncoderBlock`

**Purpose**: Stacks multiple transformer encoder layers for deep temporal feature extraction using self-attention and feedforward networks.

**Parameters**:
- `embed_dim: int` - Embedding dimension for transformer layers
- `num_heads: int = 1` - Number of attention heads per layer
- `num_layers: int = 1` - Number of stacked transformer encoder layers

**Returns**: N/A (constructor)

**Forward Pass**:
- Input: `x: Tensor` - Shape (batch, channels, timesteps)
- Output: `Tensor` - Shape (batch, channels, timesteps) after transformation

**Example**:
```python
from dlkit.core.models.nn.attention import TransformerEncoderBlock
import torch

# Create transformer encoder with 3 layers
encoder = TransformerEncoderBlock(embed_dim=128, num_heads=8, num_layers=3)

# Process temporal sequence
x = torch.randn(16, 128, 50)  # (batch, channels, timesteps)
encoded = encoder(x)  # Shape: (16, 128, 50)
```

**Implementation Notes**:
- Automatically permutes input: (batch, channels, timesteps) → (timesteps, batch, channels)
- Uses PyTorch's `nn.TransformerEncoderLayer` with default feedforward dimension
- Each layer includes self-attention + feedforward network + layer normalization
- Permutes output back to (batch, channels, timesteps)
- Default activation is GELU, default dropout is 0.0

---

### Component 3: `TransformerDecoderBlock`

**Purpose**: Stacks multiple transformer decoder layers for sequence-to-sequence tasks with optional memory (encoder output) for cross-attention.

**Parameters**:
- `embed_dim: int` - Embedding dimension for transformer layers
- `num_heads: int = 1` - Number of attention heads per layer
- `num_layers: int = 1` - Number of stacked transformer decoder layers

**Returns**: N/A (constructor)

**Forward Pass**:
- Input:
  - `x: Tensor` - Target sequence, shape (batch, channels, timesteps)
  - `memory: Tensor | None = None` - Encoder output for cross-attention (defaults to x for self-attention only)
- Output: `Tensor` - Decoded sequence, shape (batch, channels, timesteps)

**Example**:
```python
from dlkit.core.models.nn.attention import TransformerDecoderBlock
import torch

# Create transformer decoder
decoder = TransformerDecoderBlock(embed_dim=128, num_heads=8, num_layers=3)

# Self-attention mode (autoregressive)
x = torch.randn(16, 128, 50)
decoded = decoder(x)  # Uses x as both target and memory

# Cross-attention mode (with encoder output)
encoder_output = torch.randn(16, 128, 50)
target_sequence = torch.randn(16, 128, 30)
decoded = decoder(target_sequence, memory=encoder_output)  # Shape: (16, 128, 30)
```

**Implementation Notes**:
- Automatically permutes input dimensions for PyTorch transformer compatibility
- Uses PyTorch's `nn.TransformerDecoderLayer` with self-attention and cross-attention
- If memory is None, uses input sequence for both (pure self-attention)
- Includes causal masking for autoregressive generation (via PyTorch defaults)
- Each layer: self-attention + cross-attention + feedforward + layer norm

## Usage Patterns

### Common Use Case 1: Temporal Feature Enhancement with Self-Attention
```python
from dlkit.core.models.nn.attention import SelfAttentionBlock
import torch
import torch.nn as nn


class TemporalFeatureExtractor(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.attention = SelfAttentionBlock(embed_dim=channels, num_heads=num_heads)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, timesteps)
        x = self.conv(x)  # Local feature extraction
        x = self.attention(x)  # Global temporal relationships
        x = self.output(x)  # Final projection
        return x


# Usage
model = TemporalFeatureExtractor(channels=64, num_heads=8)
x = torch.randn(32, 64, 100)
output = model(x)  # Shape: (32, 64, 100)
```

### Common Use Case 2: Sequence-to-Sequence with Transformer
```python
from dlkit.core.models.nn.attention import TransformerEncoderBlock, TransformerDecoderBlock
import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 4):
        super().__init__()
        self.encoder = TransformerEncoderBlock(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
        )
        self.decoder = TransformerDecoderBlock(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Encode source sequence
        memory = self.encoder(src)
        # Decode target sequence with encoder memory
        output = self.decoder(tgt, memory=memory)
        return output


# Usage
model = Seq2SeqTransformer()
source = torch.randn(16, 128, 50)
target = torch.randn(16, 128, 30)
predictions = model(source, target)  # Shape: (16, 128, 30)
```

### Common Use Case 3: Stacked Attention Layers for Deep Temporal Processing
```python
from dlkit.core.models.nn.attention import SelfAttentionBlock
import torch
import torch.nn as nn


class DeepAttentionNetwork(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim=embed_dim, num_heads=4) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm([embed_dim, 100])  # Assuming timesteps=100
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, norm in zip(self.layers, self.norms):
            # Residual connection + layer norm
            x = norm(x + attention(x))
        return x


# Usage
model = DeepAttentionNetwork(embed_dim=64, num_layers=6)
x = torch.randn(32, 64, 100)
output = model(x)
```

## Error Handling

**Exceptions Raised**:
- `RuntimeError`: If embed_dim not divisible by num_heads
- `RuntimeError`: If input dimensions don't match embed_dim
- `ValueError`: If negative dimensions provided

**Error Handling Pattern**:
```python
from dlkit.core.models.nn.attention import SelfAttentionBlock
import torch

try:
    # This will fail: 64 not divisible by 5
    attention = SelfAttentionBlock(embed_dim=64, num_heads=5)
except RuntimeError as e:
    print(f"Invalid num_heads: {e}")
    # Use valid num_heads
    attention = SelfAttentionBlock(embed_dim=64, num_heads=8)

try:
    # This will fail: input channels (32) != embed_dim (64)
    x = torch.randn(16, 32, 100)
    output = attention(x)
except RuntimeError as e:
    print(f"Dimension mismatch: {e}")
    # Fix input dimensions
    x = torch.randn(16, 64, 100)
    output = attention(x)
```

## Testing

### Test Coverage
- Unit tests: `tests/core/models/nn/test_attention.py` (to be created)
- Integration tests: None currently

### Key Test Scenarios
1. **Self-attention with permutation**: Verify dimension handling
2. **Self-attention without permutation**: Verify pre-permuted input works
3. **Multi-head attention**: Verify multiple heads produce correct output shape
4. **Transformer encoder stacking**: Verify multiple layers compose correctly
5. **Transformer decoder cross-attention**: Verify encoder-decoder interaction
6. **Gradient flow**: Verify backpropagation through attention layers

### Fixtures Used
- Standard PyTorch tensor fixtures for various shapes
- Random seeds for reproducible attention weights

## Performance Considerations
- Attention has O(n²) complexity in sequence length - can be memory-intensive for long sequences
- Multi-head attention parallelizes well on GPUs
- Permutation operations add minimal overhead compared to attention computation
- Consider using `torch.nn.functional.scaled_dot_product_attention` for Flash Attention on newer PyTorch
- Dropout during training prevents overfitting on small datasets
- Layer normalization improves training stability

## Future Improvements / TODOs
- [ ] Add support for attention masks (for padding tokens)
- [ ] Implement Flash Attention for better memory efficiency
- [ ] Add positional encoding utilities
- [ ] Support for relative position encodings
- [ ] Causal attention mask option for autoregressive models
- [ ] Key/value caching for efficient inference
- [ ] Cross-attention only variant (no self-attention)
- [ ] Windowed attention for very long sequences
- [ ] Learnable positional embeddings
- [ ] Support for variable-length sequences with padding masks

## Related Modules
- `dlkit.core.models.nn.primitives`: Building blocks used in feedforward parts of transformers
- `dlkit.core.models.nn.encoder`: Encoder/decoder architectures that may use attention
- `dlkit.core.models.nn.cae`: Autoencoders that could benefit from attention mechanisms

## Change Log
- **2025-10-03**: Initial documentation created
