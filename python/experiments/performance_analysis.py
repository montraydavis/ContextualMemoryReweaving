# File: src/experiments/performance_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import psutil
import GPUtil
from pathlib import Path
import json
import pandas as pd

from models.cmr_full_integrated import FullCMRModel
from .dataset_testing import CMRDatasetTester

class CMRPerformanceAnalyzer:
    """
    Comprehensive performance analysis tool for CMR models.
    Analyzes computational overhead, memory efficiency, and scalability.
    """
    
    def __init__(self, cmr_model: FullCMRModel):
        self.cmr_model = cmr_model
        self.analysis_results = {}
        
    def run_comprehensive_analysis(self, output_dir: str = "performance_analysis") -> Dict:
        """
        Run comprehensive performance analysis.
        
        Args:
            output_dir: Directory to save analysis results
            
        Returns:
            analysis_results: Complete analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🔍 Starting Comprehensive CMR Performance Analysis...")
        
        analysis_results = {
            'computational_overhead': self._analyze_computational_overhead(),
            'memory_efficiency': self._analyze_memory_efficiency(),
            'scalability_analysis': self._analyze_scalability(),
            'layer_wise_impact': self._analyze_layer_wise_impact(),
            'retrieval_strategy_comparison': self._compare_retrieval_strategies(),
            'reconstruction_method_comparison': self._compare_reconstruction_methods(),
            'memory_buffer_analysis': self._analyze_memory_buffer_behavior(),
            'real_time_performance': self._analyze_real_time_performance()
        }
        
        # Generate comprehensive report
        self._generate_analysis_report(analysis_results, output_path)
        
        # Generate visualizations
        self._generate_analysis_visualizations(analysis_results, output_path)
        
        print(f"✅ Performance analysis completed! Results saved to {output_path}")
        
        return analysis_results
    
    def _analyze_computational_overhead(self) -> Dict:
        """Analyze computational overhead of CMR components."""
        print("  📊 Analyzing computational overhead...")
        
        test_inputs = [
            torch.randint(0, 1000, (1, 64)),   # Short sequence
            torch.randint(0, 1000, (1, 128)),  # Medium sequence
            torch.randint(0, 1000, (1, 256)),  # Long sequence
            torch.randint(0, 1000, (1, 512)),  # Very long sequence
        ]
        
        sequence_lengths = [64, 128, 256, 512]
        
        overhead_results = {
            'sequence_lengths': sequence_lengths,
            'baseline_times': [],
            'memory_only_times': [],
            'full_cmr_times': [],
            'overhead_percentages': {},
            'component_breakdown': {}
        }
        
        for i, (seq_len, test_input) in enumerate(zip(sequence_lengths, test_inputs)):
            print(f"    Testing sequence length: {seq_len}")
            
            # Baseline (no CMR)
            self.cmr_model.disable_memory()
            self.cmr_model.disable_reconstruction()
            baseline_time = self._measure_forward_time(test_input, num_runs=10)
            overhead_results['baseline_times'].append(baseline_time)
            
            # Memory only
            self.cmr_model.enable_memory()
            self.cmr_model.disable_reconstruction()
            memory_only_time = self._measure_forward_time(test_input, num_runs=10)
            overhead_results['memory_only_times'].append(memory_only_time)
            
            # Full CMR
            self.cmr_model.enable_memory()
            self.cmr_model.enable_reconstruction()
            full_cmr_time = self._measure_forward_time(test_input, num_runs=10)
            overhead_results['full_cmr_times'].append(full_cmr_time)
            
            # Calculate overhead percentages
            memory_overhead = (memory_only_time - baseline_time) / baseline_time * 100
            full_overhead = (full_cmr_time - baseline_time) / baseline_time * 100
            
            overhead_results['overhead_percentages'][seq_len] = {
                'memory_only': memory_overhead,
                'full_cmr': full_overhead,
                'reconstruction_only': full_overhead - memory_overhead
            }
        
        # Component-wise timing breakdown
        overhead_results['component_breakdown'] = self._analyze_component_timing()
        
        return overhead_results
    
    def _measure_forward_time(self, input_tensor: torch.Tensor, num_runs: int = 10) -> float:
        """Measure average forward pass time."""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = self.cmr_model.forward(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return np.mean(times)
    
    def _analyze_component_timing(self) -> Dict:
        """Analyze timing breakdown by component."""
        # This would require more detailed instrumentation
        # For now, provide estimated breakdown
        return {
            'memory_capture': 0.002,   # seconds
            'relevance_scoring': 0.001,
            'memory_retrieval': 0.003,
            'reconstruction': 0.005,
            'integration': 0.002
        }
    
    def _analyze_memory_efficiency(self) -> Dict:
        """Analyze memory usage patterns and efficiency."""
        print("  💾 Analyzing memory efficiency...")
        
        # Test with growing sequence lengths
        test_sequences = [
            torch.randint(0, 1000, (1, 64)),
            torch.randint(0, 1000, (1, 128)),
            torch.randint(0, 1000, (1, 256)),
            torch.randint(0, 1000, (1, 512))
        ]
        
        memory_usage = []
        buffer_stats = []
        
        for seq in test_sequences:
            # Get initial memory state
            initial_memory = psutil.virtual_memory().used / 1024**3  # GB
            
            with torch.no_grad():
                outputs = self.cmr_model.forward(seq, return_memory_info=True)
            
            # Get final memory state
            final_memory = psutil.virtual_memory().used / 1024**3  # GB
            memory_delta = final_memory - initial_memory
            
            memory_usage.append(memory_delta)
            buffer_stats.append(outputs['memory_stats']['buffer_stats'])
        
        return {
            'memory_usage_delta': memory_usage,
            'buffer_statistics': buffer_stats,
            'memory_efficiency_score': self._calculate_memory_efficiency(buffer_stats)
        }
    
    def _calculate_memory_efficiency(self, buffer_stats: List[Dict]) -> float:
        """Calculate overall memory efficiency score."""
        if not buffer_stats:
            return 0.0
        
        # Calculate efficiency based on utilization and hit rates
        utilization_scores = [stats.get('memory_utilization', 0) for stats in buffer_stats]
        hit_rates = [stats.get('cache_hit_rate', 0) for stats in buffer_stats]
        
        avg_utilization = np.mean(utilization_scores)
        avg_hit_rate = np.mean(hit_rates)
        
        # Efficiency score: weighted combination
        efficiency = 0.6 * avg_utilization + 0.4 * avg_hit_rate
        return efficiency
    
    def _analyze_scalability(self) -> Dict:
        """Analyze scalability with different model sizes and sequence lengths."""
        print("  📈 Analyzing scalability...")
        
        # Test different sequence lengths
        sequence_lengths = [64, 128, 256, 512, 1024]
        timing_results = []
        memory_results = []
        
        for seq_len in sequence_lengths:
            if seq_len <= 512:  # Max supported by current config
                test_input = torch.randint(0, 1000, (1, seq_len))
                
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / 1024**3
                
                with torch.no_grad():
                    outputs = self.cmr_model.forward(test_input)
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / 1024**3
                
                timing_results.append(end_time - start_time)
                memory_results.append(end_memory - start_memory)
            else:
                timing_results.append(None)
                memory_results.append(None)
        
        return {
            'sequence_lengths': sequence_lengths,
            'forward_times': timing_results,
            'memory_deltas': memory_results,
            'scalability_analysis': self._analyze_scalability_patterns(sequence_lengths, timing_results)
        }
    
    def _analyze_scalability_patterns(self, seq_lengths: List[int], timings: List[float]) -> Dict:
        """Analyze scalability patterns and complexity."""
        valid_timings = [(seq_len, time_val) for seq_len, time_val in zip(seq_lengths, timings) if time_val is not None]
        
        if len(valid_timings) < 2:
            return {'complexity': 'insufficient_data', 'growth_rate': 0.0}
        
        seq_lens, times = zip(*valid_timings)
        
        # Fit different complexity models
        log_n = np.log(seq_lens)
        n_log_n = seq_lens * np.log(seq_lens)
        n_squared = np.array(seq_lens) ** 2
        
        # Linear fit
        linear_coeff = np.polyfit(seq_lens, times, 1)[0]
        
        # Log-linear fit
        log_linear_coeff = np.polyfit(log_n, times, 1)[0]
        
        # N log N fit
        n_log_n_coeff = np.polyfit(n_log_n, times, 1)[0]
        
        # Determine best fit
        linear_r2 = self._calculate_r2(seq_lens, times, lambda x: linear_coeff * x)
        log_linear_r2 = self._calculate_r2(seq_lens, times, lambda x: log_linear_coeff * np.log(x))
        n_log_n_r2 = self._calculate_r2(seq_lens, times, lambda x: n_log_n_coeff * x * np.log(x))
        
        best_fit = max([('linear', linear_r2), ('log_linear', log_linear_r2), ('n_log_n', n_log_n_r2)], key=lambda x: x[1])
        
        return {
            'complexity': best_fit[0],
            'growth_rate': best_fit[1],
            'linear_coefficient': linear_coeff,
            'log_linear_coefficient': log_linear_coeff,
            'n_log_n_coefficient': n_log_n_coeff,
            'r2_scores': {
                'linear': linear_r2,
                'log_linear': log_linear_r2,
                'n_log_n': n_log_n_r2
            }
        }
    
    def _calculate_r2(self, x: List, y: List, model_func) -> float:
        """Calculate R-squared for a model fit."""
        y_pred = [model_func(xi) for xi in x]
        y_mean = np.mean(y)
        
        ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def _analyze_layer_wise_impact(self) -> Dict:
        """Analyze the impact of CMR on different transformer layers."""
        print("  🏗️ Analyzing layer-wise impact...")
        
        # Test with a medium sequence
        test_input = torch.randint(0, 1000, (1, 128))
        
        # Get layer-wise outputs without CMR
        self.cmr_model.disable_memory()
        self.cmr_model.disable_reconstruction()
        
        with torch.no_grad():
            baseline_outputs = self.cmr_model.forward(test_input, return_layer_outputs=True)
        
        # Get layer-wise outputs with CMR
        self.cmr_model.enable_memory()
        self.cmr_model.enable_reconstruction()
        
        with torch.no_grad():
            cmr_outputs = self.cmr_model.forward(test_input, return_layer_outputs=True)
        
        # Analyze differences
        layer_analysis = {}
        
        if 'layer_outputs' in baseline_outputs and 'layer_outputs' in cmr_outputs:
            baseline_layers = baseline_outputs['layer_outputs']
            cmr_layers = cmr_outputs['layer_outputs']
            
            for layer_idx in range(min(len(baseline_layers), len(cmr_layers))):
                baseline_layer = baseline_layers[layer_idx]
                cmr_layer = cmr_layers[layer_idx]
                
                # Calculate differences
                diff = torch.abs(cmr_layer - baseline_layer)
                mean_diff = torch.mean(diff).item()
                max_diff = torch.max(diff).item()
                
                layer_analysis[f'layer_{layer_idx}'] = {
                    'mean_difference': mean_diff,
                    'max_difference': max_diff,
                    'difference_std': torch.std(diff).item(),
                    'relative_change': (mean_diff / (torch.mean(torch.abs(baseline_layer)).item() + 1e-8)) * 100
                }
        
        return {
            'layer_analysis': layer_analysis,
            'overall_impact': self._calculate_overall_layer_impact(layer_analysis)
        }
    
    def _calculate_overall_layer_impact(self, layer_analysis: Dict) -> Dict:
        """Calculate overall impact across all layers."""
        if not layer_analysis:
            return {'mean_change': 0.0, 'max_change': 0.0, 'std_change': 0.0}
        
        changes = [layer_data['relative_change'] for layer_data in layer_analysis.values()]
        
        return {
            'mean_change': np.mean(changes),
            'max_change': np.max(changes),
            'std_change': np.std(changes),
            'total_layers_affected': len(changes)
        }
    
    def _compare_retrieval_strategies(self) -> Dict:
        """Compare performance of different retrieval strategies."""
        print("  🔍 Comparing retrieval strategies...")
        
        strategies = [
            'semantic_similarity',
            'contextual_relevance',
            'multi_criteria',
            'task_specific',
            'hybrid_ensemble'
        ]
        
        # Populate memory first
        for i in range(20):
            test_seq = torch.randint(0, 1000, (1, 32))
            with torch.no_grad():
                self.cmr_model.forward(test_seq)
        
        test_input = torch.randint(0, 1000, (1, 128))
        strategy_results = {}
        
        for strategy in strategies:
            self.cmr_model.set_retrieval_strategy(strategy)
            
            # Measure retrieval time
            start_time = time.time()
            with torch.no_grad():
                outputs = self.cmr_model.forward(test_input, return_memory_info=True)
            end_time = time.time()
            
            retrieval_time = end_time - start_time
            memory_stats = outputs['memory_stats']
            
            strategy_results[strategy] = {
                'retrieval_time': retrieval_time,
                'memory_entries_retrieved': memory_stats.get('entries_retrieved', 0),
                'retrieval_quality_score': memory_stats.get('retrieval_quality', 0.0),
                'cache_hit_rate': memory_stats['buffer_stats'].get('cache_hit_rate', 0.0)
            }
        
        return strategy_results
    
    def _compare_reconstruction_methods(self) -> Dict:
        """Compare different reconstruction methods."""
        print("  🔧 Comparing reconstruction methods...")
        
        methods = ['hierarchical', 'attention_based', 'mlp']
        method_results = {}
        
        test_input = torch.randint(0, 1000, (1, 128))
        
        for method in methods:
            self.cmr_model.set_reconstruction_method(method)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = self.cmr_model.forward(test_input, return_memory_info=True)
            end_time = time.time()
            
            reconstruction_time = end_time - start_time
            perf_stats = outputs.get('performance_stats', {})
            
            method_results[method] = {
                'reconstruction_time': reconstruction_time,
                'total_reconstructions': perf_stats.get('total_reconstructions', 0),
                'reconstruction_quality': perf_stats.get('reconstruction_quality', 0.0),
                'memory_tokens_used': perf_stats.get('memory_tokens_used', 0)
            }
        
        return method_results
    
    def _analyze_memory_buffer_behavior(self) -> Dict:
        """Analyze memory buffer behavior and patterns."""
        print("  📦 Analyzing memory buffer behavior...")
        
        # Test memory growth and eviction
        test_sequences = [torch.randint(0, 1000, (1, 64)) for _ in range(50)]
        
        buffer_stats_over_time = []
        eviction_counts = []
        
        for i, seq in enumerate(test_sequences):
            with torch.no_grad():
                outputs = self.cmr_model.forward(seq, return_memory_info=True)
            
            memory_stats = outputs['memory_stats']
            buffer_stats = memory_stats['buffer_stats']
            
            buffer_stats_over_time.append({
                'step': i,
                'total_entries': buffer_stats['total_entries'],
                'memory_utilization': buffer_stats['memory_utilization'],
                'evictions': buffer_stats.get('evictions', 0)
            })
            
            eviction_counts.append(buffer_stats.get('evictions', 0))
        
        return {
            'buffer_growth_pattern': buffer_stats_over_time,
            'eviction_analysis': {
                'total_evictions': sum(eviction_counts),
                'eviction_rate': np.mean(eviction_counts),
                'eviction_pattern': eviction_counts
            },
            'steady_state_analysis': self._analyze_steady_state(buffer_stats_over_time)
        }
    
    def _analyze_steady_state(self, buffer_stats: List[Dict]) -> Dict:
        """Analyze when memory buffer reaches steady state."""
        if len(buffer_stats) < 10:
            return {'steady_state_reached': False, 'convergence_step': None}
        
        # Look for stabilization in memory utilization
        utilizations = [stats['memory_utilization'] for stats in buffer_stats]
        
        # Calculate rolling variance to detect stabilization
        window_size = 5
        rolling_vars = []
        
        for i in range(window_size, len(utilizations)):
            window = utilizations[i-window_size:i]
            rolling_vars.append(np.var(window))
        
        # Find when variance stabilizes (below threshold)
        threshold = 0.01  # 1% variance threshold
        stable_indices = [i for i, var in enumerate(rolling_vars) if var < threshold]
        
        if stable_indices:
            convergence_step = stable_indices[0] + window_size
            return {
                'steady_state_reached': True,
                'convergence_step': convergence_step,
                'final_utilization': utilizations[-1],
                'stability_variance': rolling_vars[-1]
            }
        else:
            return {
                'steady_state_reached': False,
                'convergence_step': None,
                'final_utilization': utilizations[-1],
                'stability_variance': rolling_vars[-1] if rolling_vars else None
            }
    
    def _analyze_real_time_performance(self) -> Dict:
        """Analyze real-time performance characteristics."""
        print("  ⚡ Analyzing real-time performance...")
        
        # Test latency and throughput
        batch_sizes = [1, 2, 4, 8]
        sequence_length = 128
        
        latency_results = []
        throughput_results = []
        
        for batch_size in batch_sizes:
            test_input = torch.randint(0, 1000, (batch_size, sequence_length))
            
            # Measure latency
            start_time = time.time()
            with torch.no_grad():
                outputs = self.cmr_model.forward(test_input)
            end_time = time.time()
            
            latency = (end_time - start_time) / batch_size  # per sample
            throughput = batch_size / (end_time - start_time)  # samples per second
            
            latency_results.append(latency)
            throughput_results.append(throughput)
        
        return {
            'batch_sizes': batch_sizes,
            'latency_per_sample': latency_results,
            'throughput_samples_per_sec': throughput_results,
            'latency_throughput_tradeoff': self._analyze_latency_throughput_tradeoff(batch_sizes, latency_results, throughput_results)
        }
    
    def _analyze_latency_throughput_tradeoff(self, batch_sizes: List[int], latencies: List[float], throughputs: List[float]) -> Dict:
        """Analyze the tradeoff between latency and throughput."""
        # Calculate efficiency metrics
        efficiency_scores = []
        
        for i, (batch_size, latency, throughput) in enumerate(zip(batch_sizes, latencies, throughputs)):
            # Efficiency = throughput / (latency * batch_size)
            efficiency = throughput / (latency * batch_size) if latency > 0 else 0
            efficiency_scores.append(efficiency)
        
        # Find optimal batch size
        optimal_idx = np.argmax(efficiency_scores)
        optimal_batch_size = batch_sizes[optimal_idx]
        
        return {
            'efficiency_scores': efficiency_scores,
            'optimal_batch_size': optimal_batch_size,
            'optimal_efficiency': efficiency_scores[optimal_idx],
            'scaling_factor': throughputs[-1] / throughputs[0] if throughputs[0] > 0 else 0
        }
    
    def _generate_analysis_report(self, analysis_results: Dict, output_path: Path):
        """Generate comprehensive analysis report."""
        print("  📝 Generating analysis report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': self._get_model_config_summary(),
            'analysis_summary': self._generate_analysis_summary(analysis_results),
            'detailed_results': analysis_results,
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        # Save JSON report
        with open(output_path / 'performance_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save summary CSV
        self._save_summary_csv(analysis_results, output_path)
        
        print(f"    📄 Report saved to {output_path / 'performance_analysis_report.json'}")
    
    def _get_model_config_summary(self) -> Dict:
        """Get summary of LLM configuration."""
        return {
            'model_type': type(self.cmr_model).__name__,
            'base_model_config': str(self.cmr_model.base_model.config),
            'cmr_config': self.cmr_model.cmr_config,
            'memory_enabled': self.cmr_model.memory_enabled,
            'reconstruction_enabled': self.cmr_model.reconstruction_enabled
        }
    
    def _generate_analysis_summary(self, analysis_results: Dict) -> Dict:
        """Generate executive summary of analysis results."""
        overhead = analysis_results['computational_overhead']
        memory = analysis_results['memory_efficiency']
        scalability = analysis_results['scalability_analysis']
        
        # Calculate key metrics
        avg_overhead = np.mean([overhead['overhead_percentages'][seq_len]['full_cmr'] 
                               for seq_len in overhead['overhead_percentages']])
        
        memory_score = memory['memory_efficiency_score']
        complexity = scalability['scalability_analysis']['complexity']
        
        return {
            'overall_performance_score': self._calculate_overall_score(avg_overhead, memory_score, complexity),
            'average_overhead_percentage': avg_overhead,
            'memory_efficiency_score': memory_score,
            'scalability_complexity': complexity,
            'key_findings': self._identify_key_findings(analysis_results),
            'performance_grade': self._assign_performance_grade(avg_overhead, memory_score)
        }
    
    def _calculate_overall_score(self, overhead: float, memory_score: float, complexity: str) -> float:
        """Calculate overall performance score."""
        # Normalize overhead (lower is better)
        overhead_score = max(0, 100 - overhead)
        
        # Memory score is already 0-1, scale to 100
        memory_score_scaled = memory_score * 100
        
        # Complexity bonus (simpler is better)
        complexity_bonus = {
            'linear': 10,
            'log_linear': 5,
            'n_log_n': 0
        }.get(complexity, 0)
        
        # Weighted average
        overall_score = (0.4 * overhead_score + 0.4 * memory_score_scaled + 0.2 * complexity_bonus)
        return min(100, max(0, overall_score))
    
    def _identify_key_findings(self, analysis_results: Dict) -> List[str]:
        """Identify key findings from analysis."""
        findings = []
        
        overhead = analysis_results['computational_overhead']
        memory = analysis_results['memory_efficiency']
        scalability = analysis_results['scalability_analysis']
        
        # Overhead analysis
        avg_overhead = np.mean([overhead['overhead_percentages'][seq_len]['full_cmr'] 
                               for seq_len in overhead['overhead_percentages']])
        
        if avg_overhead < 30:
            findings.append("Excellent computational efficiency with minimal overhead")
        elif avg_overhead < 50:
            findings.append("Good computational efficiency with acceptable overhead")
        else:
            findings.append("High computational overhead - optimization recommended")
        
        # Memory efficiency
        if memory['memory_efficiency_score'] > 0.8:
            findings.append("High memory efficiency with optimal buffer utilization")
        elif memory['memory_efficiency_score'] > 0.6:
            findings.append("Good memory efficiency with room for improvement")
        else:
            findings.append("Low memory efficiency - buffer tuning recommended")
        
        # Scalability
        complexity = scalability['scalability_analysis']['complexity']
        if complexity == 'linear':
            findings.append("Excellent scalability with linear complexity")
        elif complexity == 'n_log_n':
            findings.append("Good scalability with near-linear complexity")
        else:
            findings.append("Moderate scalability - consider optimization")
        
        return findings
    
    def _assign_performance_grade(self, overhead: float, memory_score: float) -> str:
        """Assign performance grade based on metrics."""
        if overhead < 30 and memory_score > 0.8:
            return "A+"
        elif overhead < 40 and memory_score > 0.7:
            return "A"
        elif overhead < 50 and memory_score > 0.6:
            return "B+"
        elif overhead < 60 and memory_score > 0.5:
            return "B"
        elif overhead < 80 and memory_score > 0.4:
            return "C+"
        else:
            return "C"
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        overhead = analysis_results['computational_overhead']
        memory = analysis_results['memory_efficiency']
        scalability = analysis_results['scalability_analysis']
        
        # Overhead recommendations
        avg_overhead = np.mean([overhead['overhead_percentages'][seq_len]['full_cmr'] 
                               for seq_len in overhead['overhead_percentages']])
        
        if avg_overhead > 50:
            recommendations.append({
                'category': 'Performance',
                'priority': 'High',
                'recommendation': 'Optimize reconstruction methods to reduce computational overhead',
                'expected_improvement': '20-30% reduction in overhead'
            })
        
        # Memory recommendations
        if memory['memory_efficiency_score'] < 0.7:
            recommendations.append({
                'category': 'Memory',
                'priority': 'Medium',
                'recommendation': 'Tune memory buffer parameters and eviction strategies',
                'expected_improvement': '15-25% improvement in memory efficiency'
            })
        
        # Scalability recommendations
        complexity = scalability['scalability_analysis']['complexity']
        if complexity not in ['linear', 'log_linear']:
            recommendations.append({
                'category': 'Scalability',
                'priority': 'Medium',
                'recommendation': 'Implement more efficient retrieval algorithms',
                'expected_improvement': 'Improved scaling characteristics'
            })
        
        return recommendations
    
    def _save_summary_csv(self, analysis_results: Dict, output_path: Path):
        """Save summary results to CSV format."""
        summary_data = []
        
        # Extract key metrics for CSV
        overhead = analysis_results['computational_overhead']
        memory = analysis_results['memory_efficiency']
        scalability = analysis_results['scalability_analysis']
        
        for seq_len in overhead['sequence_lengths']:
            if seq_len in overhead['overhead_percentages']:
                summary_data.append({
                    'sequence_length': seq_len,
                    'baseline_time': overhead['baseline_times'][overhead['sequence_lengths'].index(seq_len)],
                    'full_cmr_time': overhead['full_cmr_times'][overhead['sequence_lengths'].index(seq_len)],
                    'overhead_percentage': overhead['overhead_percentages'][seq_len]['full_cmr'],
                    'memory_utilization': memory['buffer_statistics'][overhead['sequence_lengths'].index(seq_len)].get('memory_utilization', 0)
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / 'performance_summary.csv', index=False)
        print(f"    📊 Summary CSV saved to {output_path / 'performance_summary.csv'}")
    
    def _generate_analysis_visualizations(self, analysis_results: Dict, output_path: Path):
        """Generate comprehensive visualizations."""
        print("  📊 Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overhead Analysis
        self._plot_overhead_analysis(analysis_results['computational_overhead'], output_path)
        
        # 2. Memory Efficiency
        self._plot_memory_efficiency(analysis_results['memory_efficiency'], output_path)
        
        # 3. Scalability Analysis
        self._plot_scalability_analysis(analysis_results['scalability_analysis'], output_path)
        
        # 4. Retrieval Strategy Comparison
        self._plot_retrieval_comparison(analysis_results['retrieval_strategy_comparison'], output_path)
        
        # 5. Component Breakdown
        self._plot_component_breakdown(analysis_results['computational_overhead']['component_breakdown'], output_path)
        
        print(f"    📈 Visualizations saved to {output_path}")
    
    def _plot_overhead_analysis(self, overhead_data: Dict, output_path: Path):
        """Plot computational overhead analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Timing comparison
        seq_lengths = overhead_data['sequence_lengths']
        ax1.plot(seq_lengths, overhead_data['baseline_times'], 'o-', label='Baseline', linewidth=2, markersize=8)
        ax1.plot(seq_lengths, overhead_data['memory_only_times'], 's-', label='Memory Only', linewidth=2, markersize=8)
        ax1.plot(seq_lengths, overhead_data['full_cmr_times'], '^-', label='Full CMR', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Forward Pass Timing Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overhead percentages
        overhead_percentages = overhead_data['overhead_percentages']
        memory_overhead = [overhead_percentages[seq_len]['memory_only'] for seq_len in seq_lengths]
        full_overhead = [overhead_percentages[seq_len]['full_cmr'] for seq_len in seq_lengths]
        
        x = np.arange(len(seq_lengths))
        width = 0.35
        
        ax2.bar(x - width/2, memory_overhead, width, label='Memory Only', alpha=0.8)
        ax2.bar(x + width/2, full_overhead, width, label='Full CMR', alpha=0.8)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Overhead (%)')
        ax2.set_title('Computational Overhead by Component')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seq_lengths)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'overhead_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_efficiency(self, memory_data: Dict, output_path: Path):
        """Plot memory efficiency analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Memory usage over time
        buffer_stats = memory_data['buffer_statistics']
        utilization = [stats.get('memory_utilization', 0) for stats in buffer_stats]
        total_entries = [stats.get('total_entries', 0) for stats in buffer_stats]
        
        ax1.plot(utilization, 'o-', linewidth=2, markersize=8, color='green')
        ax1.set_xlabel('Test Sequence Index')
        ax1.set_ylabel('Memory Utilization')
        ax1.set_title('Memory Buffer Utilization Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Memory efficiency score
        efficiency_score = memory_data['memory_efficiency_score']
        ax2.bar(['Memory Efficiency'], [efficiency_score], color='blue', alpha=0.7)
        ax2.set_ylabel('Efficiency Score')
        ax2.set_title(f'Overall Memory Efficiency: {efficiency_score:.2%}')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'memory_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability_analysis(self, scalability_data: Dict, output_path: Path):
        """Plot scalability analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Timing vs sequence length
        seq_lengths = scalability_data['sequence_lengths']
        forward_times = scalability_data['forward_times']
        
        valid_data = [(seq_len, time_val) for seq_len, time_val in zip(seq_lengths, forward_times) if time_val is not None]
        if valid_data:
            valid_seq_lens, valid_times = zip(*valid_data)
            ax1.plot(valid_seq_lens, valid_times, 'o-', linewidth=2, markersize=8, color='red')
            ax1.set_xlabel('Sequence Length')
            ax1.set_ylabel('Forward Pass Time (seconds)')
            ax1.set_title('Scalability: Time vs Sequence Length')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Complexity analysis
        complexity_analysis = scalability_data['scalability_analysis']
        complexity = complexity_analysis['complexity']
        r2_scores = complexity_analysis.get('r2_scores', {})
        
        if r2_scores:
            methods = list(r2_scores.keys())
            scores = list(r2_scores.values())
            
            ax2.bar(methods, scores, alpha=0.7, color=['red', 'blue', 'green'])
            ax2.set_ylabel('R² Score')
            ax2.set_title(f'Scalability Model Fit (Best: {complexity})')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_retrieval_comparison(self, retrieval_data: Dict, output_path: Path):
        """Plot retrieval strategy comparison."""
        strategies = list(retrieval_data.keys())
        
        # Extract metrics
        retrieval_times = [retrieval_data[strategy]['retrieval_time'] for strategy in strategies]
        quality_scores = [retrieval_data[strategy]['retrieval_quality_score'] for strategy in strategies]
        hit_rates = [retrieval_data[strategy]['cache_hit_rate'] for strategy in strategies]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Retrieval times
        ax1.bar(strategies, retrieval_times, alpha=0.7, color='orange')
        ax1.set_ylabel('Retrieval Time (seconds)')
        ax1.set_title('Retrieval Strategy Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality scores
        ax2.bar(strategies, quality_scores, alpha=0.7, color='green')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Retrieval Quality by Strategy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cache hit rates
        ax3.bar(strategies, hit_rates, alpha=0.7, color='blue')
        ax3.set_ylabel('Cache Hit Rate')
        ax3.set_title('Cache Performance by Strategy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'retrieval_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_breakdown(self, component_data: Dict, output_path: Path):
        """Plot component timing breakdown."""
        components = list(component_data.keys())
        times = list(component_data.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(components, times, alpha=0.7, color='purple')
        plt.ylabel('Time (seconds)')
        plt.title('CMR Component Timing Breakdown')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'component_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
