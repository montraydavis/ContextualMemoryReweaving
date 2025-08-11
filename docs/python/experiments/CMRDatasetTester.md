# CMRDatasetTester

## Overview
`CMRDatasetTester` is a class in the experiments module designed for comprehensive testing of the CMR system across various datasets. It provides tools for evaluating model performance, memory behavior, and system stability under different testing scenarios.

## Key Features
- Automated test execution across multiple datasets
- Memory behavior analysis during testing
- Performance metrics collection and aggregation
- Comparative analysis between different configurations
- Visualization generation for results

## Usage
```python
from experiments.dataset_testing import CMRDatasetTester

# Initialize tester
tester = CMRDatasetTester(cmr_model, tokenizer, test_config)

# Run comprehensive tests
results = tester.run_comprehensive_tests(dataset_configs, output_dir)
```

## Supported Dataset Types
- **Conversation**: Multi-turn dialogue testing
- **Long Context**: Extended sequence processing evaluation
- **Question Answering**: QA task performance analysis
- **Summarization**: Text summarization capabilities
- **Code Generation**: Programming task evaluation

## Methods
- `run_comprehensive_tests()`: Executes all configured test cases
- `analyze_results()`: Processes and analyzes test results
- `generate_report()`: Creates detailed test reports
- `visualize_metrics()`: Generates visualizations of test metrics

## Configuration
Configure the tester using the `test_config` dictionary with parameters for test execution, metrics collection, and output generation.

## Output
Generates comprehensive reports, visualizations, and raw data for analysis of the CMR system's performance across different datasets and configurations.
