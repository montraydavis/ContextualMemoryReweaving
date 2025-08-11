# File: src/tests/test_advanced_retrieval.py
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import sys
import os

# Rely on pytest to set up import paths via conftest in the tests root

from models.advanced_retrieval import (
    RetrievalContext,
    AdvancedMemoryRetriever,
    SemanticMemoryMatcher,
    ContextualRelevanceScorer,
    MultiCriteriaRanker,
    MemoryHierarchy,
    RetrievalCache,
)
from models.memory_buffer import MemoryEntry, LayeredMemoryBuffer

class TestRetrievalContext:
    """Test the RetrievalContext dataclass."""
    
    def test_retrieval_context_creation(self):
        """Test creating a RetrievalContext instance."""
        hidden_states = torch.randn(2, 10, 128)
        context = RetrievalContext(
            current_sequence_id=1,
            current_layer_idx=5,
            current_hidden_states=hidden_states,
            attention_mask=torch.ones(2, 10),
            sequence_position=3,
            retrieval_budget=5,
            task_type="classification"
        )
        
        assert context.current_sequence_id == 1
        assert context.current_layer_idx == 5
        assert context.retrieval_budget == 5
        assert context.task_type == "classification"
        assert context.current_hidden_states.shape == (2, 10, 128)

class TestSemanticMemoryMatcher:
    """Test the SemanticMemoryMatcher class."""
    
    def test_semantic_memory_matcher_initialization(self):
        """Test SemanticMemoryMatcher initialization."""
        matcher = SemanticMemoryMatcher(hidden_size=128, similarity_threshold=0.8)
        
        assert matcher.hidden_size == 128
        assert matcher.similarity_threshold == 0.8
        assert isinstance(matcher.similarity_projector, nn.Sequential)
    
    def test_compute_similarities_empty_memories(self):
        """Test computing similarities with empty memory list."""
        matcher = SemanticMemoryMatcher(hidden_size=128)
        query_states = torch.randn(2, 10, 128)
        
        similarities = matcher.compute_similarities(query_states, [])
        assert similarities == []
    
    def test_compute_similarities_with_memories(self):
        """Test computing similarities with memory entries."""
        matcher = SemanticMemoryMatcher(hidden_size=128)
        query_states = torch.randn(2, 10, 128)
        
        # Create mock memory entries
        memory1 = Mock(spec=MemoryEntry)
        memory1.hidden_state = torch.randn(1, 128)
        memory2 = Mock(spec=MemoryEntry)
        memory2.hidden_state = torch.randn(1, 128)
        
        memories = [memory1, memory2]
        
        similarities = matcher.compute_similarities(query_states, memories)
        
        assert len(similarities) == 2
        assert all(isinstance(sim, float) for sim in similarities)
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)

class TestContextualRelevanceScorer:
    """Test the ContextualRelevanceScorer class."""
    
    def test_contextual_relevance_scorer_initialization(self):
        """Test ContextualRelevanceScorer initialization."""
        scorer = ContextualRelevanceScorer(hidden_size=128, num_heads=8)
        
        assert scorer.hidden_size == 128
        assert scorer.num_heads == 8
        assert isinstance(scorer.context_attention, nn.MultiheadAttention)
        assert isinstance(scorer.relevance_scorer, nn.Sequential)
    
    def test_contextual_relevance_scorer_adaptive_heads(self):
        """Test that num_heads is properly adjusted for small hidden_size."""
        scorer = ContextualRelevanceScorer(hidden_size=64, num_heads=8)
        
        # Should be adjusted to min(8, 64 // 16) = 4
        assert scorer.num_heads == 4
    
    def test_score_memories_empty_memories(self):
        """Test scoring with empty memory list."""
        scorer = ContextualRelevanceScorer(hidden_size=128)
        context_states = torch.randn(2, 10, 128)
        
        scores = scorer.score_memories(context_states, [])
        assert scores == []
    
    def test_score_memories_with_memories(self):
        """Test scoring with memory entries."""
        scorer = ContextualRelevanceScorer(hidden_size=128)
        context_states = torch.randn(2, 10, 128)
        
        # Create mock memory entries
        memory1 = Mock(spec=MemoryEntry)
        memory1.hidden_state = torch.randn(1, 128)
        memory2 = Mock(spec=MemoryEntry)
        memory2.hidden_state = torch.randn(1, 128)
        
        memories = [memory1, memory2]
        
        scores = scorer.score_memories(context_states, memories)
        
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        assert all(0.0 <= score <= 1.0 for score in scores)

class TestMultiCriteriaRanker:
    """Test the MultiCriteriaRanker class."""
    
    def test_multi_criteria_ranker_initialization(self):
        """Test MultiCriteriaRanker initialization."""
        criteria_weights = {
            'relevance': 0.4,
            'similarity': 0.3,
            'recency': 0.2,
            'diversity': 0.1
        }
        
        ranker = MultiCriteriaRanker(hidden_size=128, criteria_weights=criteria_weights)
        
        assert ranker.hidden_size == 128
        assert ranker.criteria_weights == criteria_weights
        assert isinstance(ranker.weight_adjuster, nn.Sequential)
    
    def test_rank_memories_empty_memories(self):
        """Test ranking with empty memory list."""
        criteria_weights = {'relevance': 0.5, 'similarity': 0.5}
        ranker = MultiCriteriaRanker(hidden_size=128, criteria_weights=criteria_weights)
        
        context = Mock(spec=RetrievalContext)
        context.current_hidden_states = torch.randn(2, 10, 128)
        
        ranked = ranker.rank_memories(context, [])
        assert ranked == []
    
    def test_rank_memories_with_memories(self):
        """Test ranking with memory entries."""
        criteria_weights = {'relevance': 0.5, 'similarity': 0.5}
        ranker = MultiCriteriaRanker(hidden_size=128, criteria_weights=criteria_weights)
        
        context = Mock(spec=RetrievalContext)
        context.current_hidden_states = torch.randn(2, 10, 128)
        
        # Create mock memory entries with relevance scores
        memory1 = Mock(spec=MemoryEntry)
        memory1.hidden_state = torch.randn(1, 128)
        memory1.relevance_score = 0.8
        memory1.timestamp = 1
        
        memory2 = Mock(spec=MemoryEntry)
        memory2.hidden_state = torch.randn(1, 128)
        memory2.relevance_score = 0.6
        memory2.timestamp = 2
        
        memories = [memory1, memory2]
        
        ranked = ranker.rank_memories(context, memories)
        
        assert len(ranked) == 2
        assert all(isinstance(memory, Mock) for memory in ranked)

class TestMemoryHierarchy:
    """Test the MemoryHierarchy class."""
    
    def test_memory_hierarchy_initialization(self):
        """Test MemoryHierarchy initialization."""
        hierarchy = MemoryHierarchy(hidden_size=128, max_clusters=16)
        
        assert hierarchy.hidden_size == 128
        assert hierarchy.max_clusters == 16
        assert hierarchy.clusters == {}
        assert hierarchy.cluster_centers == {}
        assert hierarchy.cluster_assignments == {}
    
    def test_find_relevant_clusters_empty(self):
        """Test finding relevant clusters when none exist."""
        hierarchy = MemoryHierarchy(hidden_size=128)
        query_states = torch.randn(2, 10, 128)
        
        clusters = hierarchy.find_relevant_clusters(query_states, layer_idx=5)
        assert clusters == []
    
    def test_get_cluster_memories_empty(self):
        """Test getting memories from non-existent cluster."""
        hierarchy = MemoryHierarchy(hidden_size=128)
        
        memories = hierarchy.get_cluster_memories(cluster_id=1)
        assert memories == []

class TestRetrievalCache:
    """Test the RetrievalCache class."""
    
    def test_retrieval_cache_initialization(self):
        """Test RetrievalCache initialization."""
        cache = RetrievalCache(cache_size=100)
        
        assert cache.cache_size == 100
        assert cache.cache == {}
        assert cache.access_order == []
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = RetrievalCache(cache_size=2)
        
        # Put items
        cache.put("key1", ["value1"])
        cache.put("key2", ["value2"])
        
        # Get items
        assert cache.get("key1") == ["value1"]
        assert cache.get("key2") == ["value2"]
        assert cache.get("key3") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = RetrievalCache(cache_size=2)
        
        # Fill cache
        cache.put("key1", ["value1"])
        cache.put("key2", ["value2"])
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add new item - should evict key2 (least recently used)
        cache.put("key3", ["value3"])
        
        assert "key1" in cache.cache
        assert "key2" not in cache.cache
        assert "key3" in cache.cache
    
    def test_cache_update_existing(self):
        """Test updating existing cache entry."""
        cache = RetrievalCache(cache_size=2)
        
        cache.put("key1", ["value1"])
        cache.put("key1", ["value1_updated"])
        
        assert cache.get("key1") == ["value1_updated"]
        assert len(cache.cache) == 1

class TestAdvancedMemoryRetriever:
    """Test the AdvancedMemoryRetriever class."""
    
    def test_advanced_memory_retriever_initialization(self):
        """Test AdvancedMemoryRetriever initialization."""
        # Create mock memory buffer
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}  # 12 layers
        
        retrieval_config = {
            'similarity_threshold': 0.8,
            'context_heads': 4,
            'max_clusters': 16,
            'cache_size': 500
        }
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config=retrieval_config
        )
        
        assert retriever.hidden_size == 128
        assert retriever.memory_buffer == memory_buffer
        assert retriever.config == retrieval_config
        assert isinstance(retriever.semantic_matcher, SemanticMemoryMatcher)
        assert isinstance(retriever.context_scorer, ContextualRelevanceScorer)
        assert isinstance(retriever.multi_criteria_ranker, MultiCriteriaRanker)
        assert isinstance(retriever.memory_hierarchy, MemoryHierarchy)
        assert isinstance(retriever.retrieval_cache, RetrievalCache)
    
    def test_retrieve_memories_unknown_strategy(self):
        """Test retrieval with unknown strategy raises error."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={}
        )
        
        context = Mock(spec=RetrievalContext)
        context.current_sequence_id = 1
        context.current_layer_idx = 5
        context.retrieval_budget = 5
        context.sequence_position = 3
        context.task_type = None
        context.current_hidden_states = torch.randn(2, 10, 128)
        
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            retriever.retrieve_memories(context, strategy="unknown_strategy")
    
    def test_retrieve_memories_semantic_similarity(self):
        """Test semantic similarity retrieval strategy."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        memory_buffer.retrieve_by_layer.return_value = []
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={'similarity_threshold': 0.7}
        )
        
        context = Mock(spec=RetrievalContext)
        context.current_sequence_id = 1
        context.current_layer_idx = 5
        context.retrieval_budget = 5
        context.sequence_position = 3
        context.task_type = None
        context.current_hidden_states = torch.randn(2, 10, 128)
        
        memories = retriever.retrieve_memories(context, strategy="semantic_similarity")
        assert memories == []
    
    def test_get_candidate_memories(self):
        """Test getting candidate memories from relevant layers."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        
        # Mock memories for different layers
        layer5_memories = [Mock(spec=MemoryEntry), Mock(spec=MemoryEntry)]
        layer4_memories = [Mock(spec=MemoryEntry)]
        layer6_memories = [Mock(spec=MemoryEntry)]
        
        memory_buffer.retrieve_by_layer.side_effect = lambda layer_idx: {
            4: layer4_memories,
            5: layer5_memories,
            6: layer6_memories
        }.get(layer_idx, [])
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={}
        )
        
        context = Mock(spec=RetrievalContext)
        context.current_layer_idx = 5
        
        candidate_memories = retriever._get_candidate_memories(context)
        
        # Should include memories from layers 4, 5, and 6
        expected_total = len(layer4_memories) + len(layer5_memories) + len(layer6_memories)
        assert len(candidate_memories) == expected_total
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={}
        )
        
        context = Mock(spec=RetrievalContext)
        context.current_sequence_id = 1
        context.current_layer_idx = 5
        context.sequence_position = 3
        context.retrieval_budget = 5
        context.task_type = "classification"
        
        cache_key = retriever._generate_cache_key(context, "semantic_similarity")
        
        assert cache_key.startswith("retrieval_")
        assert isinstance(cache_key, str)
    
    def test_update_retrieval_stats(self):
        """Test retrieval statistics update."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={}
        )
        
        initial_stats = retriever.retrieval_stats.copy()
        
        retriever._update_retrieval_stats("semantic_similarity", 0.1)
        
        assert retriever.retrieval_stats['total_retrievals'] == initial_stats['total_retrievals'] + 1
        assert retriever.retrieval_stats['strategy_usage']['semantic_similarity'] == 1
        assert retriever.retrieval_stats['avg_retrieval_time'] > 0
    
    def test_get_retrieval_stats(self):
        """Test getting retrieval statistics."""
        memory_buffer = Mock(spec=LayeredMemoryBuffer)
        memory_buffer.layer_buffers = {i: [] for i in range(12)}
        
        retriever = AdvancedMemoryRetriever(
            hidden_size=128,
            memory_buffer=memory_buffer,
            retrieval_config={}
        )
        
        stats = retriever.get_retrieval_stats()
        
        assert isinstance(stats, dict)
        assert 'total_retrievals' in stats
        assert 'cache_hits' in stats
        assert 'avg_retrieval_time' in stats
        assert 'strategy_usage' in stats

if __name__ == "__main__":
    pytest.main([__file__])
