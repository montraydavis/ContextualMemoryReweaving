# QueryProcessor

## Overview
`QueryProcessor` is a core service class that handles the initial processing and validation of input queries before they are used for memory retrieval. It ensures that all queries are properly formatted and ready for semantic matching.

## Key Features
- Input validation and sanitization
- Query normalization
- Feature extraction
- Error handling for malformed queries
- Integration with semantic matching services

## Usage
```python
# Example usage of QueryProcessor
from services.query_processing import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Process a query
processed_query = processor.process("example query")
```

## Methods
- `process(query)`: Main method to process and validate input queries
- `extract_features(query)`: Extracts relevant features from the query
- `validate(query)`: Validates the input query format and content
- `normalize(query)`: Normalizes the query text for consistent processing

## Configuration
Configure using the `query_processing_config` dictionary with parameters for validation rules and processing options.
