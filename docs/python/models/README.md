# Models Module

The models module contains the core neural network models and data structures for the Contextual Memory Reweaving (CMR) system. This module provides the fundamental building blocks for memory-enhanced language models.

## Overview

The models module implements the core neural network architectures and data models that enable contextual memory reweaving. It includes base transformers, specialized model variants, relevance scoring networks, and essential data structures.

## Core Components

### Base Transformer (`base_transformer.py`)

The `CMRTransformer` class provides the foundation for memory-enhanced transformers:

**Key Features:**
- Enhanced transformer with memory capture capabilities
- Mistral model integration and optimization
- Configurable hook system for state capture
- Automatic device management and quantization
- Memory tracking and sequence management

**Usage Example:**
```python
from models.base_transformer import CMRTransformer

# Initialize CMR transformer
model = CMRTransformer(config, memory_config, model_name="mistralai/Ministral-8B-Instruct-2410")

# Register memory hooks
model.register_memory_hooks()

# Process input
outputs = model(input_ids)
```

### Memory Entry (`memory_entry.py`)

The `MemoryEntry` dataclass defines the core data structure for stored memories:

**Attributes:**
- `hidden_state`: The stored tensor representation
- `layer_idx`: Transformer layer index
- `sequence_id`: Unique sequence identifier
- `position_idx`: Position within the sequence
- `relevance_score`: Computed relevance score
- `timestamp`: Creation timestamp
- `access_count`: Number of times accessed
- `last_access`: Last access timestamp

**Usage Example:**
```python
from models.memory_entry import MemoryEntry

# Create memory entry
entry = MemoryEntry(
    hidden_state=hidden_state,
    layer_idx=layer_idx,
    sequence_id=sequence_id,
    position_idx=position_idx,
    relevance_score=relevance_score,
    timestamp=time.time()
)
```

### Relevance Scorer (`relevance_scorer.py`)

The `RelevanceScorer` class determines the importance of hidden states for memory storage:

**Scoring Methods:**
- **Attention-based**: Uses attention patterns to determine relevance
- **Variance-based**: Measures activation variance across dimensions
- **Hybrid**: Combines multiple scoring approaches
- **Gradient-based**: Uses gradient information for scoring

**Usage Example:**
```python
from models.relevance_scorer import RelevanceScorer

# Initialize scorer
scorer = RelevanceScorer(hidden_size=768, scoring_method='hybrid')

# Score hidden states (module is callable)
relevance_scores = scorer(hidden_states, attention_mask)
```

Note: `FullCMRModel` defaults to `attention_based` relevance scoring, whereas `IntegratedCMRModel` defaults to `hybrid`.

### Advanced Retrieval (`advanced_retrieval.py`)

The `AdvancedMemoryRetriever` class implements sophisticated memory retrieval strategies:

**Retrieval Strategies:**
- **Semantic Similarity**: Content-based similarity matching
- **Contextual Relevance**: Context-aware relevance scoring
- **Multi-criteria**: Combines multiple ranking criteria
- **Hierarchical**: Organized memory hierarchy traversal
- **Task-specific**: Specialized retrieval for specific tasks
- **Hybrid Ensemble**: Combines multiple strategies

**Key Components:**
- `SemanticMemoryMatcher`: Semantic similarity computation
- `ContextualRelevanceScorer`: Context-aware scoring
- `MultiCriteriaRanker`: Multi-dimensional ranking
- `MemoryHierarchy`: Hierarchical memory organization
- `RetrievalCache`: Efficient caching system

**Usage Example:**
```python
from models.advanced_retrieval import AdvancedMemoryRetriever, RetrievalContext

# Initialize retriever
retriever = AdvancedMemoryRetriever(hidden_size, memory_buffer, retrieval_config)

# Create retrieval context
context = RetrievalContext(
    current_sequence_id=seq_id,
    current_layer_idx=layer_idx,
    current_hidden_states=hidden_states,
    retrieval_budget=10
)

# Retrieve memories
memories = retriever.retrieve_memories(context, strategy="multi_criteria")
```

### Performance Optimization (`performance_optimization.py`)

The `CMRPerformanceOptimizer` class provides comprehensive performance optimization:

**Optimization Components:**
- **AdaptiveThresholdManager**: Dynamic threshold adjustment
- **BatchProcessingOptimizer**: Batch size and padding optimization
- **MemoryPrefetcher**: Predictive memory loading
- **ComputationScheduler**: Computation task scheduling
- **BackgroundOptimizer**: Asynchronous optimization tasks

**Usage Example:**
```python
from models.performance_optimization import CMRPerformanceOptimizer

# Initialize optimizer
optimizer = CMRPerformanceOptimizer(cmr_model, optimization_config)

# Optimize forward pass
optimized_outputs = optimizer.optimize_forward(input_ids, attention_mask)
```

## Integrated Models

### CMR Integrated (`cmr_integrated.py`)

The `IntegratedCMRModel` orchestrates core CMR components:

**Integrated Components:**
- Base transformer with memory hooks
- Layered memory buffer
- Relevance scorer
- Hook manager

### CMR Full Integrated (`cmr_full_integrated.py`)

The `FullCMRModel` provides complete CMR implementation:

**Advanced Features:**
- Advanced memory retrieval
- State reconstruction and integration
- Performance monitoring
- Optimization layer

### Mistral Integration (`mistral_integration.py`)

The `MistralCMRModel` provides specialized Mistral integration:

**Mistral-Specific Features:**
- Optimized for Mistral architecture
- 8-bit quantization support
- Enhanced memory configuration
- Mistral-specific optimizations

## Data Models and Structures

### Memory Entry Structure

```python
@dataclass
class MemoryEntry:
    hidden_state: torch.Tensor    # Stored representation
    layer_idx: int               # Layer index
    sequence_id: int             # Sequence identifier
    position_idx: int            # Position in sequence
    relevance_score: float       # Relevance score
    timestamp: float             # Creation time
    access_count: int = 0        # Access frequency
    last_access: float = 0.0     # Last access time
```

Note: `LayeredMemoryBuffer` defines its own internal `MemoryEntry` for storage. For external usage and type references, prefer `models.memory_entry.MemoryEntry`.

### Retrieval Context Structure

```python
@dataclass
class RetrievalContext:
    current_sequence_id: int
    current_layer_idx: int
    current_hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    sequence_position: int
    retrieval_budget: int
    task_type: Optional[str]
```

## Configuration Patterns

### Model Configuration

```python
model_config = {
    'hidden_size': 768,
    'num_layers': 12,
    'num_attention_heads': 12,
    'intermediate_size': 3072,
    'max_position_embeddings': 2048
}
```

### Memory Configuration

```python
memory_config = {
    'target_layers': [6, 9, 11],
    'buffer_size': 1000,
    'relevance_threshold': 0.7,
    'scoring_method': 'hybrid',
    'max_states_per_layer': 50
}
```

### Retrieval Configuration

```python
retrieval_config = {
    'similarity_threshold': 0.7,
    'context_heads': 8,
    'max_clusters': 32,
    'cache_size': 1000,
    'criteria_weights': {
        'relevance': 0.4,
        'similarity': 0.3,
        'recency': 0.2,
        'diversity': 0.1
    }
}
```

## Performance Considerations

### Memory Management

- **Buffer Sizing**: Configure appropriate buffer sizes for your use case
- **Eviction Strategies**: Choose optimal eviction policies
- **Quantization**: Use appropriate precision for your hardware
- **Device Placement**: Optimize device allocation

### Computational Efficiency

- **Batch Processing**: Optimize batch sizes for throughput
- **Caching**: Implement effective caching strategies
- **Prefetching**: Use predictive loading for better performance
- **Background Processing**: Leverage asynchronous operations

## Best Practices

### Model Initialization

1. **Configuration Validation**: Validate all configuration parameters
2. **Device Management**: Handle device placement appropriately
3. **Memory Allocation**: Pre-allocate buffers when possible
4. **Hook Registration**: Ensure proper hook setup

### Performance Optimization

1. **Profiling**: Profile your specific use case
2. **Monitoring**: Implement comprehensive monitoring
3. **Tuning**: Iteratively tune hyperparameters
4. **Testing**: Validate performance improvements

### Error Handling

1. **Graceful Degradation**: Handle component failures
2. **Resource Monitoring**: Track resource usage
3. **Cleanup**: Implement proper cleanup procedures
4. **Logging**: Use comprehensive logging

## Troubleshooting

### Common Issues

- **Out of Memory**: Reduce buffer sizes or batch sizes
- **Slow Performance**: Enable optimization features
- **Model Loading**: Check authentication and access
- **Hook Conflicts**: Verify hook compatibility

### Performance Tips

- Use GPU acceleration when available
- Enable quantization for memory efficiency
- Configure appropriate buffer sizes
- Monitor system resources during operation
