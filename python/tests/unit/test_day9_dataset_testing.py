#!/usr/bin/env python3
"""
Tests for Day 9: Real-world Dataset Testing

This module contains comprehensive tests for the CMR dataset testing framework,
including dataset classes, metrics, and the main tester functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from experiments.dataset_testing import (
    CMRDatasetTester,
    ConversationDataset,
    LongContextDataset,
    QADataset,
    SummarizationDataset,
    CodeGenerationDataset,
    GeneralMetrics,
    ConversationMetrics,
    LongContextMetrics,
    QAMetrics,
    SummarizationMetrics,
    CodeGenerationMetrics
)

class MockCMRModel:
    """Mock CMR model for testing."""
    
    def __init__(self):
        self.device = "cpu"
        self.memory_enabled = True
        self.reconstruction_enabled = True
    
    def forward(self, input_ids, attention_mask=None, return_memory_info=False):
        batch_size, seq_len = input_ids.shape
        hidden_size = 256
        
        outputs = {
            'last_hidden_state': torch.randn(batch_size, seq_len, hidden_size),
            'hidden_states': [torch.randn(batch_size, seq_len, hidden_size) for _ in range(7)]
        }
        
        if return_memory_info:
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
            
            outputs['performance_stats'] = {
                'reconstruction_time': np.random.random() * 0.01,
                'retrieval_time': np.random.random() * 0.005,
                'total_reconstructions': np.random.randint(0, 5)
            }
        
        return outputs
    
    def enable_memory(self):
        self.memory_enabled = True
    
    def disable_memory(self):
        self.memory_enabled = False
    
    def enable_reconstruction(self):
        self.reconstruction_enabled = True
    
    def disable_reconstruction(self):
        self.reconstruction_enabled = False

class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        # Mock tokenization
        words = text.split()
        input_ids = torch.randint(0, 1000, (1, min(len(words), max_length or 512)))
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class TestDatasetClasses:
    """Test dataset classes."""
    
    def test_conversation_dataset(self):
        """Test ConversationDataset."""
        tokenizer = MockTokenizer()
        dataset = ConversationDataset("", tokenizer, max_length=256, max_samples=10)
        
        assert len(dataset) == 10
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert item['input_ids'].shape[0] <= 256
    
    def test_long_context_dataset(self):
        """Test LongContextDataset."""
        tokenizer = MockTokenizer()
        dataset = LongContextDataset("", tokenizer, max_length=1024, max_samples=5)
        
        assert len(dataset) == 5
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert item['input_ids'].shape[0] <= 1024
    
    def test_qa_dataset(self):
        """Test QADataset."""
        tokenizer = MockTokenizer()
        dataset = QADataset("", tokenizer, max_length=512, max_samples=10)
        
        assert len(dataset) == 10
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'question' in item
        assert 'context' in item
        assert 'answer' in item
    
    def test_summarization_dataset(self):
        """Test SummarizationDataset."""
        tokenizer = MockTokenizer()
        dataset = SummarizationDataset("", tokenizer, max_length=768, max_samples=8)
        
        assert len(dataset) == 8
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'text' in item
        assert 'summary' in item
    
    def test_code_generation_dataset(self):
        """Test CodeGenerationDataset."""
        tokenizer = MockTokenizer()
        dataset = CodeGenerationDataset("", tokenizer, max_length=512, max_samples=6)
        
        assert len(dataset) == 6
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'prompt' in item
        assert 'code' in item

class TestMetricsClasses:
    """Test metrics classes."""
    
    def test_general_metrics(self):
        """Test GeneralMetrics."""
        metrics = GeneralMetrics()
        
        # Mock outputs
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        
        # Test aggregation
        all_metrics = [result, result, result]
        aggregated = metrics.aggregate_metrics(all_metrics)
        assert 'perplexity' in aggregated
        assert 'score' in aggregated
    
    def test_conversation_metrics(self):
        """Test ConversationMetrics."""
        metrics = ConversationMetrics()
        
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        assert 'dialogue_coherence' in result
        assert 'response_relevance' in result
    
    def test_long_context_metrics(self):
        """Test LongContextMetrics."""
        metrics = LongContextMetrics()
        
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        assert 'context_retention' in result
        assert 'long_range_dependency' in result
    
    def test_qa_metrics(self):
        """Test QAMetrics."""
        metrics = QAMetrics()
        
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        assert 'answer_accuracy' in result
        assert 'context_understanding' in result
    
    def test_summarization_metrics(self):
        """Test SummarizationMetrics."""
        metrics = SummarizationMetrics()
        
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        assert 'summary_quality' in result
        assert 'information_retention' in result
    
    def test_code_generation_metrics(self):
        """Test CodeGenerationMetrics."""
        metrics = CodeGenerationMetrics()
        
        outputs = {'last_hidden_state': torch.randn(1, 10, 256)}
        inputs = {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        result = metrics.compute_metrics(outputs, None, inputs)
        assert 'perplexity' in result
        assert 'score' in result
        assert 'code_correctness' in result
        assert 'syntax_validity' in result

class TestCMRDatasetTester:
    """Test CMRDatasetTester."""
    
    @pytest.fixture
    def cmr_model(self):
        """Create mock CMR model."""
        return MockCMRModel()
    
    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            'enable_optimization': False,
            'optimization_config': {}
        }
    
    @pytest.fixture
    def tester(self, cmr_model, tokenizer, test_config):
        """Create CMRDatasetTester instance."""
        return CMRDatasetTester(cmr_model, tokenizer, test_config)
    
    def test_initialization(self, tester):
        """Test tester initialization."""
        assert tester.cmr_model is not None
        assert tester.tokenizer is not None
        assert tester.config is not None
        assert len(tester.dataset_loaders) == 5  # 5 dataset types
    
    def test_dataset_loading(self, tester):
        """Test dataset loading functionality."""
        # Test conversation dataset loading
        config = {
            'data_path': '',
            'max_length': 256,
            'max_samples': 5
        }
        
        dataset = tester._load_conversation_dataset(config)
        assert isinstance(dataset, ConversationDataset)
        assert len(dataset) == 5
        
        # Test long context dataset loading
        dataset = tester._load_long_context_dataset(config)
        assert isinstance(dataset, LongContextDataset)
    
    def test_metrics_collector_selection(self, tester):
        """Test metrics collector selection."""
        # Test each dataset type
        assert isinstance(tester._get_metrics_collector('conversation'), ConversationMetrics)
        assert isinstance(tester._get_metrics_collector('long_context'), LongContextMetrics)
        assert isinstance(tester._get_metrics_collector('question_answering'), QAMetrics)
        assert isinstance(tester._get_metrics_collector('summarization'), SummarizationMetrics)
        assert isinstance(tester._get_metrics_collector('code_generation'), CodeGenerationMetrics)
        assert isinstance(tester._get_metrics_collector('unknown'), GeneralMetrics)
    
    def test_memory_behavior_analysis(self, tester):
        """Test memory behavior analysis."""
        # Mock memory stats
        memory_stats = [
            {
                'buffer_stats': {'total_entries': 10},
                'layer_stats': {f'layer_{i}': {'entries': 5} for i in range(12)}
            },
            {
                'buffer_stats': {'total_entries': 15},
                'layer_stats': {f'layer_{i}': {'entries': 8} for i in range(12)}
            }
        ]
        
        analysis = tester._analyze_memory_behavior(memory_stats)
        
        assert 'total_entries' in analysis
        assert 'layer_distribution' in analysis
        assert 'memory_efficiency' in analysis
        assert analysis['total_entries']['mean'] == 12.5
        assert analysis['total_entries']['growth_rate'] == 5.0
    
    def test_performance_behavior_analysis(self, tester):
        """Test performance behavior analysis."""
        # Mock performance stats
        performance_stats = [
            {
                'reconstruction_time': 0.01,
                'retrieval_time': 0.005,
                'total_reconstructions': 2
            },
            {
                'reconstruction_time': 0.015,
                'retrieval_time': 0.008,
                'total_reconstructions': 3
            }
        ]
        
        analysis = tester._analyze_performance_behavior(performance_stats)
        
        assert 'timing' in analysis
        assert 'reconstruction' in analysis
        assert analysis['reconstruction']['total_reconstructions'] == 5
        assert analysis['reconstruction']['avg_reconstructions_per_batch'] == 2.5
    
    def test_comparative_analysis(self, tester):
        """Test comparative analysis."""
        # Mock dataset results
        dataset_results = {
            'dataset1': {
                'metrics': {'perplexity': 5.0, 'score': 0.8},
                'memory_analysis': {
                    'memory_efficiency': {'avg_entries_per_batch': 20}
                }
            },
            'dataset2': {
                'metrics': {'perplexity': 3.0, 'score': 0.9},
                'memory_analysis': {
                    'memory_efficiency': {'avg_entries_per_batch': 15}
                }
            }
        }
        
        comparative = tester._comparative_analysis(dataset_results)
        
        assert 'performance_ranking' in comparative
        assert 'memory_efficiency_ranking' in comparative
        
        # Check performance ranking (lower perplexity = higher score)
        perf_ranking = comparative['performance_ranking']
        assert len(perf_ranking) == 2
        assert perf_ranking[0]['dataset'] == 'dataset2'  # Lower perplexity
    
    def test_performance_summary_generation(self, tester):
        """Test performance summary generation."""
        # Mock dataset results
        dataset_results = {
            'dataset1': {
                'metrics': {'perplexity': 5.0, 'score': 0.8},
                'memory_analysis': {
                    'memory_efficiency': {'avg_entries_per_batch': 20}
                }
            },
            'dataset2': {
                'metrics': {'perplexity': 3.0, 'score': 0.9},
                'memory_analysis': {
                    'memory_efficiency': {'avg_entries_per_batch': 15}
                }
            }
        }
        
        # Set test_results for the tester
        tester.test_results = {
            'comparative_analysis': {
                'performance_ranking': [{'dataset': 'dataset2', 'score': 0.9}],
                'memory_efficiency_ranking': [{'dataset': 'dataset2', 'score': 0.8}]
            }
        }
        
        summary = tester._generate_performance_summary(dataset_results)
        
        assert summary['total_datasets'] == 2
        assert summary['successful_tests'] == 2
        assert summary['failed_tests'] == 0
        assert summary['best_performing_dataset'] == 'dataset2'
        assert summary['most_memory_efficient'] == 'dataset2'
    
    def test_overall_metrics_aggregation(self, tester):
        """Test overall metrics aggregation."""
        all_metrics = [
            {'perplexity': 5.0, 'score': 0.8},
            {'perplexity': 3.0, 'score': 0.9},
            {'perplexity': 4.0, 'score': 0.85}
        ]
        
        aggregated = tester._aggregate_overall_metrics(all_metrics)
        
        assert 'perplexity' in aggregated
        assert 'score' in aggregated
        assert aggregated['perplexity']['mean'] == 4.0
        assert np.isclose(aggregated['score']['mean'], 0.85, rtol=1e-10)
    
    def test_save_results(self, tester):
        """Test results saving functionality."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test data with numpy types
            test_results = {
                'perplexity': np.float32(5.0),
                'score': np.float64(0.8),
                'array_data': np.array([1, 2, 3]),
                'nested': {
                    'value': np.int32(42)
                }
            }
            
            # Save results
            filepath = temp_path / "test_results.json"
            tester._save_results(test_results, filepath)
            
            # Verify file was created
            assert filepath.exists()
            
            # Load and verify content
            with open(filepath, 'r') as f:
                loaded_results = json.load(f)
            
            # Check that numpy types were converted
            assert isinstance(loaded_results['perplexity'], float)
            assert isinstance(loaded_results['score'], float)
            assert isinstance(loaded_results['array_data'], list)
            assert isinstance(loaded_results['nested']['value'], int)
    
    def test_comprehensive_testing(self, tester):
        """Test comprehensive testing workflow."""
        # Create test dataset configurations
        dataset_configs = [
            {
                'name': 'test_conversation',
                'type': 'conversation',
                'max_length': 256,
                'max_samples': 5,
                'batch_size': 2,
                'test_config': {
                    'enable_memory': True,
                    'enable_reconstruction': True
                }
            }
        ]
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run comprehensive tests
            results = tester.run_comprehensive_tests(dataset_configs, temp_dir)
            
            # Verify results structure
            assert 'dataset_results' in results
            assert 'comparative_analysis' in results
            assert 'performance_summary' in results
            
            # Verify dataset results
            dataset_results = results['dataset_results']
            assert 'test_conversation' in dataset_results
            
            # Check that results were saved
            temp_path = Path(temp_dir)
            assert (temp_path / "test_conversation_results.json").exists()
            assert (temp_path / "comprehensive_results.json").exists()
    
    def test_error_handling(self, tester):
        """Test error handling in dataset testing."""
        # Test with invalid dataset type
        invalid_config = {
            'name': 'invalid_test',
            'type': 'invalid_type',
            'max_length': 256,
            'max_samples': 5
        }
        
        # This should raise an error
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            tester._load_dataset('invalid_type', invalid_config)
    
    def test_optimization_impact_analysis(self, tester):
        """Test optimization impact analysis."""
        # Test without optimizer
        impact = tester._analyze_optimization_impact()
        assert impact == {}
        
        # Test with optimizer (would need to mock optimizer)
        # This is a placeholder test for now
        assert True

class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow."""
        # Create all components
        cmr_model = MockCMRModel()
        tokenizer = MockTokenizer()
        test_config = {'enable_optimization': False}
        
        tester = CMRDatasetTester(cmr_model, tokenizer, test_config)
        
        # Create minimal test configuration
        dataset_configs = [
            {
                'name': 'integration_test',
                'type': 'conversation',
                'max_length': 128,
                'max_samples': 3,
                'batch_size': 1,
                'test_config': {
                    'enable_memory': True,
                    'enable_reconstruction': True
                }
            }
        ]
        
        # Run test
        with tempfile.TemporaryDirectory() as temp_dir:
            results = tester.run_comprehensive_tests(dataset_configs, temp_dir)
            
            # Verify basic structure
            assert 'dataset_results' in results
            assert 'integration_test' in results['dataset_results']
            
            # Verify no errors
            dataset_result = results['dataset_results']['integration_test']
            assert 'error' not in dataset_result
            assert dataset_result['status'] != 'failed'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
