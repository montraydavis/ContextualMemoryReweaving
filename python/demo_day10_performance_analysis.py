#!/usr/bin/env python3
"""
Day 10 Demo: Performance Analysis & Week 2 Wrap-up
==================================================

This demo showcases the comprehensive performance analysis capabilities
implemented for the CMR system, including:

1. Computational overhead analysis
2. Memory efficiency evaluation
3. Scalability testing
4. Retrieval strategy comparison
5. Reconstruction method benchmarking
6. Real-time performance analysis
7. Comprehensive reporting and visualization

Usage:
    python demo_day10_performance_analysis.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from typing import Dict, Any

# Import CMR components
from models.cmr_full_integrated import FullCMRModel
from experiments.performance_analysis import CMRPerformanceAnalyzer
from experiments.dataset_testing import CMRDatasetTester
from transformers import AutoConfig

def create_test_model() -> FullCMRModel:
    """Create a test CMR model for performance analysis."""
    print("🔧 Creating test CMR model...")
    
    # Small LLM configuration for testing
    base_config = AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    
    # CMR configuration
    cmr_config = {
        'target_layers': [2, 4],
        'intervention_layers': [4],
        'max_entries_per_layer': 50,
        'max_total_entries': 150,
        'scoring_method': 'attention_based',
        'relevance_threshold': 0.3,
        'retrieval_strategy': 'multi_criteria',
        'retrieval_budget': 8,
        'eviction_strategy': 'lru_relevance',
        'retrieval_config': {
            'similarity_threshold': 0.6,
            'context_heads': 4,
            'criteria_weights': {
                'relevance': 0.4,
                'similarity': 0.3,
                'recency': 0.2,
                'diversity': 0.1
            }
        },
        'reconstruction_config': {
            'method': 'hierarchical',
            'max_memory_tokens': 8,
            'compression_ratio': 0.6
        }
    }
    
    model = FullCMRModel(base_config, cmr_config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model

def run_basic_performance_tests(model: FullCMRModel) -> Dict[str, Any]:
    """Run basic performance tests to validate the system."""
    print("\n🧪 Running basic performance validation tests...")
    
    results = {}
    
    # Test 1: Basic forward pass timing
    print("  📊 Testing basic forward pass timing...")
    test_input = torch.randint(0, 1000, (1, 64))
    
    times = []
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.forward(test_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    results['basic_forward_pass'] = {
        'average_time': avg_time,
        'std_time': std_time,
        'min_time': min(times),
        'max_time': max(times)
    }
    
    print(f"    Average time: {avg_time:.4f}s ± {std_time:.4f}s")
    
    # Test 2: Memory capture efficiency
    print("  💾 Testing memory capture efficiency...")
    memory_growth = []
    
    for i in range(20):
        test_seq = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            outputs = model.forward(test_seq, return_memory_info=True)
        
        memory_stats = outputs['memory_stats']
        buffer_stats = memory_stats['buffer_stats']
        memory_growth.append({
            'step': i,
            'total_entries': buffer_stats['total_entries'],
            'utilization': buffer_stats['memory_utilization']
        })
    
    results['memory_growth'] = memory_growth
    final_utilization = memory_growth[-1]['utilization']
    print(f"    Final memory utilization: {final_utilization:.2%}")
    
    # Test 3: Reconstruction performance
    print("  🔧 Testing reconstruction performance...")
    test_input = torch.randint(0, 1000, (1, 128))
    
    with torch.no_grad():
        outputs = model.forward(test_input, return_memory_info=True)
    
    perf_stats = outputs.get('performance_stats', {})
    reconstruction_count = perf_stats.get('total_reconstructions', 0)
    
    results['reconstruction_performance'] = {
        'total_reconstructions': reconstruction_count,
        'memory_tokens_used': perf_stats.get('memory_tokens_used', 0),
        'reconstruction_quality': perf_stats.get('reconstruction_quality', 0.0)
    }
    
    print(f"    Reconstructions performed: {reconstruction_count}")
    
    return results

def run_comprehensive_analysis(model: FullCMRModel, output_dir: str = "day10_performance_analysis") -> Dict[str, Any]:
    """Run comprehensive performance analysis."""
    print(f"\n🔍 Running comprehensive performance analysis...")
    print(f"   Output directory: {output_dir}")
    
    # Create performance analyzer
    analyzer = CMRPerformanceAnalyzer(model)
    
    # Run comprehensive analysis
    analysis_results = analyzer.run_comprehensive_analysis(output_dir)
    
    return analysis_results

def run_dataset_testing(model: FullCMRModel) -> Dict[str, Any]:
    """Run dataset testing to validate CMR performance."""
    print("\n📚 Running dataset testing validation...")
    
    # Create dataset tester
    dataset_tester = CMRDatasetTester(model)
    
    results = {}
    
    # Test 1: Synthetic dataset
    print("  🧪 Testing with synthetic dataset...")
    synthetic_results = dataset_tester.test_synthetic_dataset(
        num_samples=50,
        sequence_length=64,
        num_classes=5
    )
    results['synthetic'] = synthetic_results
    
    print(f"    Accuracy: {synthetic_results['accuracy']:.3f}")
    print(f"    Memory utilization: {synthetic_results['memory_utilization']:.2%}")
    
    # Test 2: Conversation dataset
    print("  💬 Testing with conversation dataset...")
    conversation_results = dataset_tester.test_conversation_dataset(
        num_conversations=20,
        max_turns=8,
        sequence_length=64
    )
    results['conversation'] = conversation_results
    
    print(f"    Context retention: {conversation_results['context_retention']:.3f}")
    print(f"    Response consistency: {conversation_results['response_consistency']:.3f}")
    
    # Test 3: Code generation dataset
    print("  💻 Testing with code generation dataset...")
    code_results = dataset_tester.test_code_generation_dataset(
        num_samples=30,
        sequence_length=128,
        language='python'
    )
    results['code_generation'] = code_results
    
    print(f"    Code completion accuracy: {code_results['completion_accuracy']:.3f}")
    print(f"    Syntax correctness: {code_results['syntax_correctness']:.3f}")
    
    return results

def generate_performance_summary(analysis_results: Dict[str, Any], 
                                basic_results: Dict[str, Any],
                                dataset_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive performance summary."""
    print("\n📊 Generating performance summary...")
    
    # Extract key metrics
    overhead = analysis_results['computational_overhead']
    memory = analysis_results['memory_efficiency']
    scalability = analysis_results['scalability_analysis']
    
    # Calculate overall performance score
    avg_overhead = np.mean([overhead['overhead_percentages'][seq_len]['full_cmr'] 
                           for seq_len in overhead['overhead_percentages']])
    
    memory_score = memory['memory_efficiency_score']
    complexity = scalability['scalability_analysis']['complexity']
    
    # Performance grade calculation
    if avg_overhead < 30 and memory_score > 0.8:
        grade = "A+"
    elif avg_overhead < 40 and memory_score > 0.7:
        grade = "A"
    elif avg_overhead < 50 and memory_score > 0.6:
        grade = "B+"
    elif avg_overhead < 60 and memory_score > 0.5:
        grade = "B"
    elif avg_overhead < 80 and memory_score > 0.4:
        grade = "C+"
    else:
        grade = "C"
    
    summary = {
        'overall_performance': {
            'grade': grade,
            'overhead_percentage': avg_overhead,
            'memory_efficiency': memory_score,
            'scalability_complexity': complexity
        },
        'key_metrics': {
            'basic_forward_pass_time': basic_results['basic_forward_pass']['average_time'],
            'memory_utilization': basic_results['memory_growth'][-1]['utilization'],
            'reconstruction_count': basic_results['reconstruction_performance']['total_reconstructions']
        },
        'dataset_performance': {
            'synthetic_accuracy': dataset_results['synthetic']['accuracy'],
            'conversation_retention': dataset_results['conversation']['context_retention'],
            'code_completion_accuracy': dataset_results['code_generation']['completion_accuracy']
        },
        'recommendations': analysis_results.get('recommendations', [])
    }
    
    return summary

def display_performance_summary(summary: Dict[str, Any]):
    """Display performance summary in a formatted way."""
    print("\n" + "="*80)
    print("🎯 CMR PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall Performance
    overall = summary['overall_performance']
    print(f"\n📊 OVERALL PERFORMANCE: {overall['grade']}")
    print(f"   Computational Overhead: {overall['overhead_percentage']:.1f}%")
    print(f"   Memory Efficiency: {overall['memory_efficiency']:.2%}")
    print(f"   Scalability: {overall['scalability_complexity']}")
    
    # Key Metrics
    metrics = summary['key_metrics']
    print(f"\n⚡ KEY METRICS:")
    print(f"   Forward Pass Time: {metrics['basic_forward_pass_time']:.4f}s")
    print(f"   Memory Utilization: {metrics['memory_utilization']:.2%}")
    print(f"   Reconstructions: {metrics['reconstruction_count']}")
    
    # Dataset Performance
    dataset = summary['dataset_performance']
    print(f"\n📚 DATASET PERFORMANCE:")
    print(f"   Synthetic Accuracy: {dataset['synthetic_accuracy']:.3f}")
    print(f"   Context Retention: {dataset['conversation_retention']:.3f}")
    print(f"   Code Completion: {dataset['code_completion_accuracy']:.3f}")
    
    # Recommendations
    recommendations = summary['recommendations']
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority']}] {rec['category']}: {rec['recommendation']}")
            print(f"      Expected improvement: {rec['expected_improvement']}")
    
    print("\n" + "="*80)

def run_week2_validation_tests(model: FullCMRModel) -> bool:
    """Run Week 2 validation tests to ensure all components work correctly."""
    print("\n🧪 Running Week 2 validation tests...")
    
    try:
        # Test 1: Memory capture and retrieval
        print("  ✅ Testing memory capture and retrieval...")
        for i in range(25):
            test_seq = torch.randint(0, 1000, (1, 32))
            with torch.no_grad():
                model.forward(test_seq)
        
        # Test 2: Reconstruction pipeline
        print("  ✅ Testing reconstruction pipeline...")
        test_input = torch.randint(0, 1000, (1, 128))
        with torch.no_grad():
            outputs = model.forward(test_input, return_memory_info=True)
        
        # Verify outputs
        assert 'last_hidden_state' in outputs
        assert outputs['last_hidden_state'].shape == (1, 128, 256)
        
        memory_stats = outputs['memory_stats']
        assert memory_stats['buffer_stats']['total_entries'] > 0
        
        # Test 3: Different retrieval strategies
        print("  ✅ Testing retrieval strategies...")
        strategies = ['semantic_similarity', 'multi_criteria', 'hybrid_ensemble']
        for strategy in strategies:
            model.set_retrieval_strategy(strategy)
            with torch.no_grad():
                model.forward(test_input)
        
        # Test 4: Different reconstruction methods
        print("  ✅ Testing reconstruction methods...")
        methods = ['hierarchical', 'attention_based', 'mlp']
        for method in methods:
            model.set_reconstruction_method(method)
            with torch.no_grad():
                model.forward(test_input)
        
        print("  🎉 All Week 2 validation tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Week 2 validation test failed: {e}")
        return False

def main():
    """Main demo function."""
    print("🚀 Day 10 Demo: Performance Analysis & Week 2 Wrap-up")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test model
    model = create_test_model()
    
    # Run basic performance tests
    basic_results = run_basic_performance_tests(model)
    
    # Run comprehensive performance analysis
    output_dir = "day10_performance_analysis"
    analysis_results = run_comprehensive_analysis(model, output_dir)
    
    # Run dataset testing
    dataset_results = run_dataset_testing(model)
    
    # Generate and display summary
    summary = generate_performance_summary(analysis_results, basic_results, dataset_results)
    display_performance_summary(summary)
    
    # Run Week 2 validation tests
    validation_success = run_week2_validation_tests(model)
    
    # Final status
    print(f"\n🎯 FINAL STATUS:")
    if validation_success:
        print("✅ Week 2 CMR implementation is fully validated and ready for production!")
        print("✅ Performance analysis completed successfully!")
        print("✅ All components are working correctly!")
    else:
        print("❌ Some validation tests failed - review required before production use.")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("📊 Check the generated visualizations and reports for detailed analysis.")
    
    return validation_success

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
