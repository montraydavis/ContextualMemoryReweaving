#!/usr/bin/env python3
"""
Demo script for Day 9: Real-world Dataset Testing

This script demonstrates the comprehensive testing framework for CMR models
on various real-world datasets including conversation, long context, QA,
summarization, and code generation tasks.
"""

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer
import tempfile
import shutil
from pathlib import Path
import json

from models.cmr_full_integrated import FullCMRModel
from experiments.dataset_testing import CMRDatasetTester

def create_mock_cmr_model():
    """Create a mock CMR model for testing purposes."""
    # Small LLM configuration for testing
    llm_model_config = AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    
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
    
    # Create model
    model = FullCMRModel(llm_model_config, cmr_config, device="cpu")
    
    # Mock the forward method to return expected outputs
    def mock_forward(input_ids, attention_mask=None, return_memory_info=False):
        batch_size, seq_len = input_ids.shape
        hidden_size = getattr(llm_model_config, 'hidden_size', 256)
        
        # Mock transformer outputs
        outputs = {
            'last_hidden_state': torch.randn(batch_size, seq_len, hidden_size),
            'hidden_states': [torch.randn(batch_size, seq_len, hidden_size) for _ in range(getattr(llm_model_config, 'num_hidden_layers', 6) + 1)]
        }
        
        if return_memory_info:
            # Mock memory stats
            outputs['memory_stats'] = {
                'buffer_stats': {
                    'total_entries': np.random.randint(10, 100),
                    'layer_distribution': {f'layer_{i}': np.random.randint(5, 20) for i in range(12)}
                },
                'layer_stats': {
                    f'layer_{i}': {
                        'entries': np.random.randint(5, 20),
                        'utilization': np.random.random()
                    } for i in range(12)
                }
            }
            
            # Mock performance stats
            outputs['performance_stats'] = {
                'reconstruction_time': np.random.random() * 0.01,
                'retrieval_time': np.random.random() * 0.005,
                'total_reconstructions': np.random.randint(0, 5)
            }
        
        return outputs
    
    # Replace the forward method
    model.forward = mock_forward
    
    # Mock enable/disable methods
    def mock_enable_memory():
        model.memory_enabled = True
    
    def mock_disable_memory():
        model.memory_enabled = False
    
    def mock_enable_reconstruction():
        model.reconstruction_enabled = True
    
    def mock_disable_reconstruction():
        model.reconstruction_enabled = False
    
    model.enable_memory = mock_enable_memory
    model.disable_memory = mock_disable_memory
    model.enable_reconstruction = mock_enable_reconstruction
    model.disable_reconstruction = mock_disable_reconstruction
    
    return model

def create_test_configs():
    """Create test configurations for different dataset types."""
    return [
        {
            'name': 'conversation_test',
            'type': 'conversation',
            'max_length': 512,
            'max_samples': 50,
            'batch_size': 4,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': True
            }
        },
        {
            'name': 'long_context_test',
            'type': 'long_context',
            'max_length': 1024,
            'max_samples': 25,
            'batch_size': 2,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': True
            }
        },
        {
            'name': 'qa_test',
            'type': 'question_answering',
            'max_length': 512,
            'max_samples': 50,
            'batch_size': 4,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': False
            }
        },
        {
            'name': 'summarization_test',
            'type': 'summarization',
            'max_length': 768,
            'max_samples': 40,
            'batch_size': 3,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': True
            }
        },
        {
            'name': 'code_generation_test',
            'type': 'code_generation',
            'max_length': 512,
            'max_samples': 30,
            'batch_size': 2,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': True
            }
        }
    ]

def run_dataset_testing_demo():
    """Run the main dataset testing demonstration."""
    print("🚀 Day 9: Real-world Dataset Testing Demo")
    print("=" * 60)
    
    # Create temporary directory for results
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Create mock CMR model
        print("\n🔧 Creating mock CMR model...")
        cmr_model = create_mock_cmr_model()
        print("   ✅ Mock CMR model created successfully")
        
        # Create tokenizer
        print("\n🔤 Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Ministral-8B-Instruct-2410')
        tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer created successfully")
        
        # Create dataset tester
        print("\n🧪 Initializing dataset tester...")
        test_config = {
            'enable_optimization': False,  # Disable for demo
            'optimization_config': {}
        }
        
        tester = CMRDatasetTester(cmr_model, tokenizer, test_config)
        print("   ✅ Dataset tester initialized successfully")
        
        # Create test configurations
        print("\n📋 Creating test configurations...")
        dataset_configs = create_test_configs()
        print(f"   ✅ Created {len(dataset_configs)} test configurations")
        
        # Run comprehensive tests
        print("\n🚀 Starting comprehensive dataset testing...")
        results = tester.run_comprehensive_tests(dataset_configs, str(temp_dir))
        
        # Display results summary
        print("\n📊 Testing Results Summary")
        print("-" * 40)
        
        summary = results.get('performance_summary', {})
        print(f"Total datasets tested: {summary.get('total_datasets', 0)}")
        print(f"Successful tests: {summary.get('successful_tests', 0)}")
        print(f"Failed tests: {summary.get('failed_tests', 0)}")
        
        if summary.get('best_performing_dataset'):
            print(f"Best performing dataset: {summary['best_performing_dataset']}")
        
        if summary.get('most_memory_efficient'):
            print(f"Most memory efficient: {summary['most_memory_efficient']}")
        
        # Display comparative analysis
        print("\n🔍 Comparative Analysis")
        print("-" * 40)
        
        comparative = results.get('comparative_analysis', {})
        
        # Performance ranking
        if 'performance_ranking' in comparative:
            print("\nPerformance Ranking:")
            for i, item in enumerate(comparative['performance_ranking'][:3]):
                print(f"  {i+1}. {item['dataset']}: {item['score']:.4f}")
        
        # Memory efficiency ranking
        if 'memory_efficiency_ranking' in comparative:
            print("\nMemory Efficiency Ranking:")
            for i, item in enumerate(comparative['memory_efficiency_ranking'][:3]):
                print(f"  {i+1}. {item['dataset']}: {item['score']:.4f}")
        
        # Display individual dataset results
        print("\n📈 Individual Dataset Results")
        print("-" * 40)
        
        for dataset_name, dataset_result in results.get('dataset_results', {}).items():
            if 'error' not in dataset_result:
                print(f"\n{dataset_name}:")
                print(f"  Type: {dataset_result.get('dataset_type', 'Unknown')}")
                print(f"  Status: ✅ Success")
                
                # Display metrics
                metrics = dataset_result.get('metrics', {})
                if metrics:
                    print(f"  Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.4f}")
                        else:
                            print(f"    {key}: {value}")
                
                # Display memory analysis
                memory_analysis = dataset_result.get('memory_analysis', {})
                if memory_analysis:
                    total_entries = memory_analysis.get('total_entries', {})
                    if 'mean' in total_entries:
                        print(f"  Memory: {total_entries['mean']:.1f} avg entries")
                
                # Display performance analysis
                performance_analysis = dataset_result.get('performance_analysis', {})
                if performance_analysis:
                    reconstruction = performance_analysis.get('reconstruction', {})
                    if 'total_reconstructions' in reconstruction:
                        print(f"  Reconstructions: {reconstruction['total_reconstructions']}")
            else:
                print(f"\n{dataset_name}:")
                print(f"  Status: ❌ Failed")
                print(f"  Error: {dataset_result['error']}")
        
        # Show generated files
        print(f"\n📁 Generated Files in {temp_dir}:")
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                print(f"  📄 {file_path.name} ({file_size} bytes)")
        
        print(f"\n🎉 Dataset testing demo completed successfully!")
        print(f"   All results saved to: {temp_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")

def run_quick_test():
    """Run a quick test with minimal data."""
    print("🚀 Quick Dataset Testing Demo")
    print("=" * 40)
    
    # Create minimal test configs
    quick_configs = [
        {
            'name': 'quick_conversation',
            'type': 'conversation',
            'max_length': 256,
            'max_samples': 10,
            'batch_size': 2,
            'test_config': {
                'enable_memory': True,
                'enable_reconstruction': True
            }
        }
    ]
    
    # Create mock model and tester
    cmr_model = create_mock_cmr_model()
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Ministral-8B-Instruct-2410')
    tokenizer.pad_token = tokenizer.eos_token
    
    tester = CMRDatasetTester(cmr_model, tokenizer, {'enable_optimization': False})
    
    # Run quick test
    results = tester.run_comprehensive_tests(quick_configs, "quick_test_results")
    
    print(f"\n✅ Quick test completed!")
    return results

if __name__ == "__main__":
    print("🎯 CMR Day 9: Real-world Dataset Testing")
    print("=" * 60)
    
    # Check if user wants quick test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_test()
    else:
        run_dataset_testing_demo()
    
    print("\n�� Demo completed!")
