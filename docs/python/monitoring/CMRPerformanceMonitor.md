# CMRPerformanceMonitor

## Overview
`CMRPerformanceMonitor` is the central class for monitoring and collecting performance metrics in the CMR system. It provides real-time insights into system health, performance characteristics, and operational metrics.

## Key Features
- Comprehensive metric collection
- Real-time performance monitoring
- Health status assessment
- Alerting and notifications
- Integration with visualization tools

## Usage
```python
from monitoring.performance_monitoring import CMRPerformanceMonitor

# Initialize monitor with configuration
monitor = CMRPerformanceMonitor(
    config={
        'enable_real_time': True,
        'metric_collection_interval': 1.0,
        'health_check_interval': 5.0
    }
)

# Track a metric
monitor.track_metric('inference_latency', 0.125)

# Get system health status
health_status = monitor.get_health_status()

# Generate performance report
report = monitor.generate_report()
```

## Core Components
- **Metric Collection**: System and application metrics
- **Health Monitoring**: Component health status
- **Alerting System**: Threshold-based notifications
- **Performance Analysis**: Trend and pattern analysis
- **Resource Tracking**: CPU, GPU, and memory utilization

## Methods
- `track_metric(name, value, timestamp=None)`: Records a metric value
- `get_health_status()`: Returns current system health status
- `generate_report(time_range='1h')`: Generates a performance report
- `set_alert(metric_name, condition, callback)`: Configures alert conditions
- `reset_metrics()`: Clears all collected metrics

## Configuration
Configure using the `monitoring_config` dictionary with parameters for collection intervals, alert thresholds, and storage backends.
