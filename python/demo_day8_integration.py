#!/usr/bin/env python3
"""
Demo script for Day 8: Integration with Base Model
Demonstrates the full CMR integration with GPT-OSS 20B base model.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
import time
import numpy as np
from typing import Dict, List, Optional

from models.cmr_full_integrated import FullCMRModel
from models.performance_optimization import CMRPerformanceOptimizer

def create_cmr_config() -> Dict:
    """Create configuration for CMR model."""
    return {
        'max_entries_per_layer': 500,
        'max_total_entries': 2000,
        'eviction_strategy': 'lru_relevance',
        'scoring_method': 'attention_based',
        'target_layers': [4, 8, 11],
        'intervention_layers': [4, 8, 11],
        'relevance_threshold': 0.2,  # Lower threshold to capture more memories
        'retrieval_strategy': 'multi_criteria',
        'retrieval_config': {
            'max_memories': 10,
            'similarity_threshold': 0.7,
            'use_semantic_search': True
        },
        'reconstruction_config': {
            'integration_method': 'weighted_sum',
            'memory_weight': 0.3,
            'layer_weights': {
                4: 0.4,   # Lower layers get more memory influence
                8: 0.3,   # Middle layers
                11: 0.2   # Higher layers get less memory influence
            }
        }
    }

def create_optimization_config() -> Dict:
    """Create configuration for performance optimization."""
    return {
        'enable_background_optimization': True,
        'enable_batch_optimization': True,
        'enable_prefetching': True,
        'enable_adaptive_thresholds': True,
        'optimization_interval': 15.0  # seconds
    }

def demonstrate_memory_capture_and_reconstruction():
    """Demonstrate memory capture and reconstruction capabilities."""
    print("🔧 Day 8: CMR Integration with Base Model")
    print("=" * 60)
    
    # Create GPT-OSS 20B configuration
    gpt2_config = AutoConfig.from_pretrained("openai-community/gpt2")
    
    # Create CMR configuration
    cmr_config = create_cmr_config()
    
    # Initialize the full CMR model
    print("Initializing Full CMR Model...")
    cmr_model = FullCMRModel(
        base_config=gpt2_config,
        cmr_config=cmr_config,
        device="cpu"  # Use CPU for demo
    )
    
    # Initialize performance optimizer
    print("Initializing Performance Optimizer...")
    optimization_config = create_optimization_config()
    optimizer = CMRPerformanceOptimizer(cmr_model, optimization_config)
    
    # Create tokenizer for demo
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Demo sequences
    demo_sequences = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Context memory reweaving combines transformer architectures with external memory systems.",
        "The neural network processes information through multiple layers of computation.",
        "Memory systems enhance model performance by storing and retrieving relevant information."
    ]
    
    print("\n📚 Demo Sequences:")
    for i, seq in enumerate(demo_sequences):
        print(f"  {i+1}. {seq}")
    
    print("\n🔄 Running Memory Capture and Reconstruction Demo...")
    print("-" * 60)
    
    # Process sequences to build memory
    for i, sequence in enumerate(demo_sequences):
        print(f"\nProcessing Sequence {i+1}:")
        print(f"  Text: {sequence[:50]}...")
        
        # Tokenize
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Optimize forward pass
        start_time = time.time()
        opt_inputs, opt_mask, opt_info = optimizer.optimize_forward_pass(
            inputs['input_ids'],
            inputs['attention_mask']
        )
        optimization_time = time.time() - start_time
        
        # Forward pass with memory capture
        forward_start = time.time()
        outputs = cmr_model(
            input_ids=opt_inputs,
            attention_mask=opt_mask,
            task_type="text_processing",
            sequence_metadata={"demo_sequence": i+1},
            return_memory_info=True
        )
        forward_time = time.time() - forward_start
        
        # Display results
        print(f"  Optimization: {optimization_time:.4f}s")
        print(f"  Forward Pass: {forward_time:.4f}s")
        print(f"  Total Time: {optimization_time + forward_time:.4f}s")
        
        if opt_info.get('batch_optimized'):
            print(f"  Batch Optimized: {opt_info['original_shape']} → {opt_info['optimized_shape']}")
        
        if opt_info.get('threshold_adjusted'):
            print(f"  Threshold Adjusted: {opt_info['threshold_adjusted']:.3f}")
        
        # Show memory stats
        memory_stats = outputs.get('memory_stats', {})
        if memory_stats:
            buffer_stats = memory_stats.get('buffer_stats', {})
            total_entries = buffer_stats.get('total_entries', 0)
            print(f"  Memory Entries: {total_entries}")
    
    print("\n📊 Final Memory Statistics:")
    print("-" * 60)
    
    # Get comprehensive statistics
    memory_usage = cmr_model.get_memory_usage()
    performance_stats = cmr_model.performance_monitor.get_stats()
    optimization_stats = optimizer.get_optimization_stats()
    
    print(f"Memory Usage:")
    print(f"  Total Entries: {memory_usage.get('total_entries', 0)}")
    print(f"  Entries per Layer: {memory_usage.get('entries_per_layer', {})}")
    print(f"  Memory Size: {memory_usage.get('memory_size_mb', 0):.2f} MB")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Captures: {performance_stats.get('total_captures', 0)}")
    print(f"  Total Reconstructions: {performance_stats.get('total_reconstructions', 0)}")
    print(f"  States Stored per Layer: {performance_stats.get('states_stored_per_layer', {})}")
    
    print(f"\nOptimization Statistics:")
    print(f"  Threshold Adjustments: {optimization_stats.get('threshold_adjustments', 0)}")
    print(f"  Batch Optimizations: {optimization_stats.get('batch_optimizations', 0)}")
    print(f"  Prefetch Hits: {optimization_stats.get('prefetch_stats', {}).get('hit_count', 0)}")
    
    # Demonstrate memory retrieval
    print("\n🔍 Memory Retrieval Demo:")
    print("-" * 60)
    
    # Create a query sequence
    query_text = "The fox and the dog are animals that can jump and run."
    print(f"Query: {query_text}")
    
    query_inputs = tokenizer(
        query_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    
    # Process query with memory retrieval
    query_outputs = cmr_model(
        input_ids=query_inputs['input_ids'],
        attention_mask=query_inputs['attention_mask'],
        task_type="memory_query",
        return_memory_info=True
    )
    
    # Show retrieval statistics
    retrieval_stats = query_outputs.get('retrieval_stats', {})
    if retrieval_stats:
        print(f"Memories Retrieved: {retrieval_stats.get('memories_retrieved', 0)}")
        print(f"Retrieval Time: {retrieval_stats.get('avg_retrieval_time', 0):.4f}s")
    
    # Demonstrate memory clearing
    print("\n🧹 Memory Management Demo:")
    print("-" * 60)
    
    print("Clearing all memories...")
    cmr_model.clear_memory()
    
    # Verify memory is cleared
    memory_usage_after = cmr_model.get_memory_usage()
    print(f"Memory after clearing: {memory_usage_after.get('total_entries', 0)} entries")
    
    # Stop background optimizer
    print("\nStopping background optimizer...")
    optimizer.background_optimizer.stop()
    
    print("\n✅ Day 8 Demo Completed Successfully!")
    print("=" * 60)

def demonstrate_advanced_features():
    """Demonstrate advanced CMR features."""
    print("\n🚀 Advanced Features Demo:")
    print("-" * 60)
    
    # Create a smaller model for faster demo
    gpt2_config = GPT2Config(
        n_layer=6,
        n_head=8,
        n_embd=512,
        vocab_size=50257,
        max_position_embeddings=512
    )
    
    cmr_config = create_cmr_config()
    cmr_config['target_layers'] = [2, 4, 5]  # Fewer layers for demo
    cmr_config['intervention_layers'] = [2, 4, 5]
    
    cmr_model = FullCMRModel(
        base_config=gpt2_config,
        cmr_config=cmr_config,
        device="cpu"
    )
    
    # Demonstrate adaptive threshold management
    print("1. Adaptive Threshold Management:")
    
    # Simulate different memory usage scenarios
    test_sequences = [
        "Short text for testing.",
        "This is a medium length sequence that will test the adaptive threshold system.",
        "This is a very long sequence that will test how the system adapts to different input lengths and memory pressure conditions."
    ]
    
    for i, seq in enumerate(test_sequences):
        print(f"  Testing sequence {i+1} (length: {len(seq)})")
        
        # Process sequence
        inputs = torch.randint(0, 1000, (1, len(seq.split())))
        attention_mask = torch.ones_like(inputs)
        
        outputs = cmr_model(
            input_ids=inputs,
            attention_mask=attention_mask,
            return_memory_info=True
        )
        
        # Show current threshold
        current_threshold = cmr_model.relevance_threshold
        print(f"    Current threshold: {current_threshold:.3f}")
    
    # Demonstrate memory buffer statistics
    print("\n2. Memory Buffer Statistics:")
    buffer_stats = cmr_model.memory_buffer.get_statistics()
    print(f"  Total entries: {buffer_stats.get('total_entries', 0)}")
    print(f"  Entries per layer: {buffer_stats.get('entries_per_layer', {})}")
    print(f"  Memory size: {buffer_stats.get('memory_size_mb', 0):.2f} MB")
    
    # Demonstrate performance monitoring
    print("\n3. Performance Monitoring:")
    perf_stats = cmr_model.performance_monitor.get_stats()
    print(f"  Total captures: {perf_stats.get('total_captures', 0)}")
    print(f"  Total reconstructions: {perf_stats.get('total_reconstructions', 0)}")
    
    if perf_stats.get('avg_capture_time'):
        print(f"  Average capture time: {perf_stats['avg_capture_time']:.6f}s")
    
    if perf_stats.get('avg_reconstruction_time'):
        print(f"  Average reconstruction time: {perf_stats['avg_reconstruction_time']:.6f}s")
    
    print("\n✅ Advanced Features Demo Completed!")

if __name__ == "__main__":
    try:
        # Main demo
        demonstrate_memory_capture_and_reconstruction()
        
        # Advanced features demo
        demonstrate_advanced_features()
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
