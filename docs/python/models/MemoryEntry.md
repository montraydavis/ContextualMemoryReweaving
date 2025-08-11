# MemoryEntry

## Overview

`MemoryEntry` is a data class that represents a single memory unit in the CMR system. It encapsulates the hidden states captured from transformer layers along with relevant metadata for memory retrieval and management.

## Key Features

- Stores hidden states with associated metadata
- Tracks relevance scores for memory prioritization
- Supports flexible metadata attachment
- Implements memory serialization

## Class Structure

```python
@dataclass
class MemoryEntry:
    hidden_state: torch.Tensor  # The captured hidden state
    layer_idx: int             # Source layer index
    sequence_id: int           # Identifier for the sequence
    relevance_score: float     # Calculated relevance score
    metadata: dict = field(default_factory=dict)  # Additional metadata
```

## Usage Example

```python
from models.memory import MemoryEntry
import torch

# Create a new memory entry
memory = MemoryEntry(
    hidden_state=torch.randn(1, 32, 768),  # Example hidden state
    layer_idx=4,
    sequence_id=42,
    relevance_score=0.85,
    metadata={
        'timestamp': '2025-08-11T03:42:00',
        'source': 'user_input',
        'tags': ['important', 'context']
    }
)

# Access memory properties
print(f"Layer: {memory.layer_idx}")
print(f"Relevance: {memory.relevance_score:.2f}")
print(f"Metadata: {memory.metadata}")
```

## Attributes

- `hidden_state`: The captured hidden state tensor
- `layer_idx`: Index of the layer where the state was captured
- `sequence_id`: Identifier for the input sequence
- `relevance_score`: Score indicating memory importance (0.0 to 1.0)
- `metadata`: Dictionary for additional contextual information

## Methods

- `__post_init__()`: Validates and processes the memory entry after initialization
- `to_dict()`: Serializes the memory entry to a dictionary
- `from_dict(data)`: Deserializes a dictionary back to a MemoryEntry

## Best Practices

- Keep metadata lightweight to minimize memory overhead
- Use consistent sequence IDs for related memories
- Update relevance scores based on usage patterns
- Implement custom serialization for complex metadata types
