# CMRPerformanceAnalyzer

## Overview
`CMRPerformanceAnalyzer` is a class in the experiments module that provides detailed performance analysis capabilities for the CMR system. It helps in understanding the computational and memory characteristics of the system under different workloads and configurations.

## Key Features
- Computational overhead analysis (CPU/GPU usage)
- Memory efficiency evaluation
- Scalability analysis across sequence lengths
- Memory buffer behavior analysis
- Real-time performance monitoring

## Usage
```python
from experiments.performance_analysis import CMRPerformanceAnalyzer

# Initialize analyzer
analyzer = CMRPerformanceAnalyzer(cmr_model)

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis(output_dir)
```

## Key Metrics
- **Forward Pass Timing**: Measures model inference time
- **Memory Usage**: Tracks memory consumption patterns
- **Retrieval Efficiency**: Analyzes memory retrieval performance
- **Reconstruction Quality**: Evaluates state reconstruction accuracy
- **Cache Performance**: Monitors cache hit/miss rates
- **Eviction Patterns**: Tracks memory buffer eviction behavior

## Methods
- `run_comprehensive_analysis()`: Executes all performance tests
- `analyze_memory_usage()`: Detailed memory consumption analysis
- `measure_throughput()`: Calculates processing throughput
- `generate_performance_report()`: Creates detailed performance reports
- `visualize_metrics()`: Generates visualizations of performance data

## Configuration
The analyzer can be configured with various parameters to control the depth and scope of performance analysis, including test durations, sequence lengths, and resource monitoring settings.

## Output
Generates performance reports, visualizations, and raw metric data that help in understanding the system's performance characteristics and identifying potential bottlenecks.
