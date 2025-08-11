# FullCMRModel

## Overview
`FullCMRModel` is the main integration class that provides end-to-end functionality for the Contextual Memory Reweaving (CMR) system. It serves as the primary interface for using CMR capabilities with various language models.

## Key Features
- Complete CMR system integration
- Memory capture and storage
- Context-aware memory retrieval
- State reconstruction
- Performance monitoring
- Configuration management

## Usage
```python
from models.cmr_full_integrated import FullCMRModel

# Initialize with base and CMR configurations
model = FullCMRModel(base_config, cmr_config)

# Process input with memory capabilities
outputs = model.forward(input_ids, return_memory_info=True)
```

## Management API
- `enable_memory(True/False)`: Toggle memory functionality
- `enable_reconstruction(True/False)`: Toggle state reconstruction
- `set_retrieval_strategy(name)`: Configure retrieval approach
- `set_reconstruction_method(name)`: Set reconstruction technique
- `optimize_memory()`: Optimize memory buffer
- `save_memory(path)`: Persist memory state
- `load_memory(path)`: Restore memory state

## Configuration
Configure using `base_config` for the base model and `cmr_config` for CMR-specific settings including:
- Memory buffer size
- Retrieval strategies
- Reconstruction methods
- Performance optimization flags

## Integration Components
- Base Transformer model
- LayeredMemoryBuffer for storage
- RelevanceScorer for memory retrieval
- LayeredStateReconstructor for state integration
- Performance monitoring system

## Best Practices
- Initialize with appropriate configuration for your use case
- Monitor memory usage for large-scale deployments
- Use the management API to optimize performance
- Regularly save memory state for persistence
