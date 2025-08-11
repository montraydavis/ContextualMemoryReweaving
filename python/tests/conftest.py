"""
Shared pytest fixtures for CMR tests.
"""
import pytest
import torch
from transformers import AutoConfig
from models.memory_buffer import LayeredMemoryBuffer


@pytest.fixture
def small_config():
    """Create small configuration for testing."""
    return AutoConfig.from_pretrained("openai-community/gpt2")


@pytest.fixture
def medium_config():
    """Create medium configuration for testing."""
    return AutoConfig.from_pretrained("openai-community/gpt2")


@pytest.fixture
def basic_memory_config():
    """Create basic memory configuration."""
    return {
        'target_layers': [2, 4],
        'scoring_method': 'hybrid',
        'relevance_threshold': 0.3,
        'max_entries_per_layer': 50,
        'max_total_entries': 200,
        'eviction_strategy': 'lru_relevance',
        'memory_retrieval_k': 5
    }


@pytest.fixture
def small_buffer():
    """Create small memory buffer for testing."""
    return LayeredMemoryBuffer(
        max_entries_per_layer=5,
        max_total_entries=20,
        eviction_strategy="lru_relevance"
    )


@pytest.fixture
def medium_buffer():
    """Create medium memory buffer for testing."""
    return LayeredMemoryBuffer(
        max_entries_per_layer=10,
        max_total_entries=50,
        eviction_strategy="lru_relevance"
    )


@pytest.fixture
def hidden_size():
    """Standard hidden size for testing."""
    return 768


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def seq_len():
    """Standard sequence length for testing."""
    return 50


@pytest.fixture
def hidden_states(batch_size, seq_len, hidden_size):
    """Create sample hidden states tensor."""
    return torch.randn(batch_size, seq_len, hidden_size)


@pytest.fixture
def attention_mask(batch_size, seq_len):
    """Create sample attention mask with padding."""
    mask = torch.ones(batch_size, seq_len)
    mask[0, 45:] = 0  # Mask last 5 positions of first sequence
    return mask
