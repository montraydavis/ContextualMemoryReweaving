# RelevanceScorer

## Overview
`RelevanceScorer` is a service class that calculates relevance scores for memory items based on various criteria. It helps determine the most pertinent memories for a given context.

## Key Features
- Multi-criteria scoring
- Contextual relevance calculation
- Configurable weighting of different factors
- Integration with semantic matching
- Performance optimization for high-throughput scenarios

## Usage
```python
from services.relevance_scoring import RelevanceScorer

# Initialize with configuration
scorer = RelevanceScorer(weights={'semantic': 0.6, 'temporal': 0.3, 'frequency': 0.1})

# Score memories
scores = scorer.score_memories(query_embedding, memories)
```

## Methods
- `score_memories(query_embedding, memories)`: Scores a list of memories
- `calculate_relevance(query_embedding, memory_embedding)`: Calculates relevance score
- `update_weights(new_weights)`: Updates scoring weights
- `normalize_scores(scores)`: Normalizes scores to a consistent range

## Configuration
Configure using the `relevance_scoring_config` dictionary with parameters for different scoring factors and their respective weights.
