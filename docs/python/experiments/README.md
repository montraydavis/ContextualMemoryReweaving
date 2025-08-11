# Experiments Module

The experiments module contains comprehensive testing and analysis frameworks for the Contextual Memory Reweaving (CMR) system. This module provides tools for evaluating CMR performance across different datasets, analyzing computational overhead, and benchmarking various system configurations.

## Overview

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Demo Scripts](#demo-scripts)
- [Configuration](#configuration)
- [Output and Results](#output-and-results)
- [Integration with CMR System](#integration-with-cmr-system)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

```mermaid
graph TD
    %% Main Experiment Flow
    A[Test Configuration] -->|Load| B[Test Runner]
    B -->|Validate| C[Dataset Loading]
    C -->|Preprocess| D[Model Execution]
    D -->|Collect| E[Metric Collection]
    E -->|Analyze| F[Analysis & Reporting]
    F -->|Generate| G[Visualization]
    
    %% Error Handling
    A -->|Invalid Config| H[Error Handler]
    C -->|Load Failed| H
    D -->|Runtime Error| H
    E -->|Data Issue| H
    H -->|Log| I[Error Report]
    
    %% Data Validation
    subgraph "Validation Steps"
        VA[Config Validation] -->|Check Params| VB[Resource Check]
        VB -->|Sufficient| VC[Dataset Validation]
        VC -->|Format Check| VD[Schema Validation]
    end
    
    A --> VA
    VD -->|Valid| B
    
    %% Performance Metrics
    subgraph "Performance Monitoring"
        M1[Load Time]
        M2[Execution Time]
        M3[Memory Usage]
        M4[GPU Utilization]
        M5[Throughput]
    end
    
    C -->|Time| M1
    D -->|Time| M2
    D -->|Memory| M3
    D -->|GPU Stats| M4
    E -->|Samples/s| M5
    
    %% Component Interactions
    subgraph "Experiment Pipeline"
        B <-->|Control| C
        C <-->|Cache| D
        D <-->|Profile| E
        E <-->|Feedback| F
        F <-->|Export| G
    end
    
    %% Visualization Types
    subgraph "Visualization Types"
        G1[Performance Charts]
        G2[Memory Plots]
        G3[Accuracy Metrics]
        G4[Comparison Tables]
    end
    
    G --> G1 & G2 & G3 & G4
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#9f9,stroke:#333,stroke-width:2px
    style I fill:#f99,stroke:#333,stroke-width:2px
    
    %% Legend
    subgraph " "
        direction TB
        L1[Process]:::process
        L2[Validation]:::validation
        L3[Metrics]:::metrics
        L4[Error]:::error
        L5[Output]:::output
    end
    
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef validation fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef metrics fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    classDef output fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
```

The experiments module is designed to validate and analyze the CMR system's performance in real-world scenarios. It includes dataset testing capabilities, performance analysis tools, and comprehensive benchmarking frameworks.

### Experiment Workflow

1. **Test Configuration**: Define experiment parameters and test cases
2. **Test Runner**: Orchestrates the execution of experiments
3. **Dataset Loading**: Loads and preprocesses test datasets
4. **Model Execution**: Runs CMR model with test inputs
5. **Metric Collection**: Gathers performance and accuracy metrics
6. **Analysis & Reporting**: Processes and analyzes results
7. **Visualization**: Generates visual reports and comparisons

## Key Components

### Dataset Testing (`dataset_testing.py`)

[View Class Documentation](./dataset_testing.md)

The `CMRDatasetTester` class provides comprehensive testing capabilities across multiple dataset types:

**Supported Dataset Types:**

- **Conversation**: Multi-turn dialogue testing
- **Long Context**: Extended sequence processing evaluation
- **Question Answering**: QA task performance analysis
- **Summarization**: Text summarization capabilities
- **Code Generation**: Programming task evaluation

**Key Features:**

- Automated test execution across multiple datasets
- Memory behavior analysis during testing
- Performance metrics collection and aggregation
- Comparative analysis between different configurations
- Visualization generation for results

**Usage Example:**

```python
from experiments.dataset_testing import CMRDatasetTester

# Initialize tester
tester = CMRDatasetTester(cmr_model, tokenizer, test_config)

# Run comprehensive tests
results = tester.run_comprehensive_tests(dataset_configs, output_dir)
```

### Performance Analysis (`performance_analysis.py`)

[View Class Documentation](./performance_analysis.md)

The `CMRPerformanceAnalyzer` class provides detailed performance analysis capabilities:

**Analysis Types:**

- **Computational Overhead**: CPU/GPU usage analysis
- **Memory Efficiency**: Memory consumption patterns
- **Scalability Analysis**: Performance across different sequence lengths
- **Layer-wise Impact**: Per-layer performance breakdown
- **Retrieval Strategy Comparison**: Benchmarking different retrieval methods
- **Reconstruction Method Comparison**: Evaluating reconstruction approaches
- **Memory Buffer Analysis**: Buffer behavior and efficiency
- **Real-time Performance**: Live performance monitoring

**Key Metrics:**

- Forward pass timing
- Memory usage patterns
- Retrieval efficiency
- Reconstruction quality
- Cache performance: cache_hits and cache_misses (rate calculation planned)
- Eviction patterns

**Usage Example:**

```python
from experiments.performance_analysis import CMRPerformanceAnalyzer

# Initialize analyzer
analyzer = CMRPerformanceAnalyzer(cmr_model)

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis(output_dir)
```

## Demo Scripts

### Dataset Testing Demo (`demo_day9_dataset_testing.py`)

Demonstrates the dataset testing framework with:

- Mock dataset creation
- Multi-dataset testing execution
- Results analysis and visualization
- Performance metrics collection

### Performance Analysis Demo (`demo_day10_performance_analysis.py`)

Showcases performance analysis capabilities including:

- Computational overhead analysis
- Memory efficiency evaluation
- Scalability testing
- Strategy comparison benchmarks

## Configuration

### Test Configuration Structure

```python
test_config = {
    'enable_optimization': True,
    'optimization_config': {
        'enable_prefetching': True,
        'enable_batch_optimization': True
    }
}
```

### Dataset Configuration Structure

```python
dataset_config = {
    'name': 'conversation_test',
    'type': 'conversation',
    'max_length': 512,
    'max_samples': 50,
    'batch_size': 4,
    'test_config': {
        'enable_memory': True,
        'enable_reconstruction': True
    }
}
```

## Output and Results

### Generated Files

The experiments module generates comprehensive output including:

- **JSON Reports**: Detailed metrics and analysis results
- **Visualizations**: Performance charts and trend analysis
- **CSV Data**: Raw metrics for further analysis
- **Summary Reports**: High-level findings and recommendations
- **Performance Summary CSV**: `performance_summary.csv` generated by the analyzer

### Metrics Collected

- **Performance Metrics**: Timing, throughput, latency
- **Memory Metrics**: Usage patterns, efficiency, eviction rates
- **Quality Metrics**: Reconstruction quality, retrieval accuracy
- **System Metrics**: Resource utilization, cache performance

## Integration with CMR System

The experiments module integrates seamlessly with:

- **FullCMRModel**: Complete system testing
- **Memory Buffer**: Buffer behavior analysis
- **Retrieval System**: Strategy evaluation
- **Reconstruction System**: Method comparison
- **Performance Optimization**: Optimization impact analysis

## Best Practices

1. **Test Configuration**: Use appropriate batch sizes and sequence lengths for your hardware
2. **Dataset Selection**: Choose representative datasets for your use case
3. **Metric Interpretation**: Focus on metrics most relevant to your application
4. **Resource Management**: Monitor system resources during long-running tests
5. **Result Analysis**: Compare results across different configurations systematically

## Troubleshooting

### Common Issues

- **Memory Errors**: Reduce batch size or sequence length
- **Slow Performance**: Enable optimization features
- **Missing Dependencies**: Ensure all required packages are installed
- **Dataset Loading**: Verify dataset paths and formats

### Performance Tips

- Use GPU acceleration when available
- Enable batch optimization for better throughput
- Configure appropriate memory limits
- Use caching for repeated experiments
