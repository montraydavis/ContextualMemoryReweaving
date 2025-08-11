# SemanticMatcher

## Overview

`SemanticMatcher` is a service class that handles semantic matching of queries against stored memories. It uses advanced NLP techniques to find contextually relevant matches beyond simple keyword matching.

## Key Features

- Semantic similarity scoring
- Context-aware matching
- Multi-dimensional vector search
- Integration with embedding models
- Caching for performance optimization

## Usage

```python
from services.semantic_matching import SemanticMatcher

# Initialize with optional configuration
matcher = SemanticMatcher(embedding_model='all-mpnet-base-v2')

# Find similar memories
matches = matcher.find_similar(query_embedding, memories, top_k=5)
```

## Methods

- `find_similar(query_embedding, memories, top_k=5)`: Finds top-k similar memories
- `calculate_similarity(embedding1, embedding2)`: Calculates similarity score
- `batch_match(queries, memories)`: Processes multiple queries efficiently
- `update_embedding_model(model_name)`: Updates the embedding model

## Configuration

Configure using the `semantic_matching_config` dictionary with parameters for model selection, similarity thresholds, and caching options.
