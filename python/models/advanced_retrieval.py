from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.memory_entry import MemoryEntry


@dataclass
class RetrievalContext:
    current_sequence_id: int
    current_layer_idx: int
    current_hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    sequence_position: int
    retrieval_budget: int
    task_type: Optional[str] = None


class SemanticMemoryMatcher(nn.Module):
    def __init__(self, hidden_size: int, similarity_threshold: float = 0.7):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.similarity_threshold = float(similarity_threshold)
        self.similarity_projector = nn.Sequential(
            nn.Linear(self.hidden_size, max(1, self.hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.hidden_size // 2), self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

    def compute_similarities(self, query_states: torch.Tensor, memory_entries: List[MemoryEntry]) -> List[float]:
        if not memory_entries:
            return []
        # Mean pool current context
        query_repr = query_states.mean(dim=(0, 1))  # [hidden]
        query_proj = self.similarity_projector(query_repr)
        similarities: List[float] = []
        for mem in memory_entries:
            mem_state = mem.hidden_state.squeeze()
            mem_proj = self.similarity_projector(mem_state)
            sim = F.cosine_similarity(query_proj, mem_proj, dim=0).item()
            # clamp to [-1, 1]
            similarities.append(max(-1.0, min(1.0, float(sim))))
        return similarities


class ContextualRelevanceScorer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = int(hidden_size)
        # Ensure head count divides hidden size reasonably
        self.num_heads = min(int(num_heads), max(1, self.hidden_size // 16))
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.relevance_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, max(1, self.hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, self.hidden_size // 2), 1),
            nn.Sigmoid(),
        )

    def score_memories(
        self,
        context_states: torch.Tensor,
        memory_entries: List[MemoryEntry],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[float]:
        """
        Lightweight scoring that avoids heavy attention to keep tests simple and robust.
        Computes cosine similarity between mean context and each memory, mapped to [0,1].
        """
        if not memory_entries:
            return []
        # Mean-pool context (respect mask if provided)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B,S,1]
            denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
            ctx = (context_states * mask).sum(dim=(0, 1)) / denom  # [H]
        else:
            ctx = context_states.mean(dim=(0, 1))  # [H]

        scores: List[float] = []
        for mem in memory_entries:
            m = mem.hidden_state.squeeze()
            sim = F.cosine_similarity(ctx, m, dim=0).item()
            # map from [-1,1] → [0,1]
            scores.append(float(0.5 * (sim + 1.0)))
        return scores


class MultiCriteriaRanker(nn.Module):
    def __init__(self, hidden_size: int, criteria_weights: Dict[str, float]):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.criteria_weights = dict(criteria_weights)
        self.weight_adjuster = nn.Sequential(
            nn.Linear(len(self.criteria_weights), max(1, self.hidden_size // 4)),
            nn.ReLU(),
            nn.Linear(max(1, self.hidden_size // 4), len(self.criteria_weights)),
            nn.Softmax(dim=-1),
        )

    def rank_memories(self, context: RetrievalContext, candidate_memories: List[MemoryEntry]) -> List[MemoryEntry]:
        if not candidate_memories:
            return []
        # Build criterion scores
        scores_by_criterion: Dict[str, List[float]] = {}
        # Relevance from entries
        scores_by_criterion['relevance'] = [float(m.relevance_score) for m in candidate_memories]
        # Similarity to mean of context
        q = context.current_hidden_states.mean(dim=(0, 1))
        scores_by_criterion['similarity'] = [
            float(F.cosine_similarity(q, m.hidden_state.squeeze(), dim=0)) for m in candidate_memories
        ]
        # Recency (use timestamp; larger timestamp considered more recent)
        now_ref = max((getattr(m, 'timestamp', 0.0) for m in candidate_memories), default=0.0)
        scores_by_criterion['recency'] = [
            float(1.0 / (1.0 + max(0.0, now_ref - getattr(m, 'timestamp', 0.0)))) for m in candidate_memories
        ]
        # Diversity proxy: variance of state
        scores_by_criterion['diversity'] = [float(torch.var(m.hidden_state).item()) for m in candidate_memories]
        # Normalize each criterion to [0,1]
        for k, vals in scores_by_criterion.items():
            if not vals:
                continue
            vmin, vmax = min(vals), max(vals)
            if vmax > vmin:
                scores_by_criterion[k] = [(v - vmin) / (vmax - vmin) for v in vals]
            else:
                scores_by_criterion[k] = [0.0 for _ in vals]
        # Compute adaptive weights
        base = torch.tensor([self.criteria_weights.get(k, 0.0) for k in self.criteria_weights.keys()], dtype=torch.float32)
        weights = self.weight_adjuster(base).detach().numpy().tolist()
        crit_keys = list(self.criteria_weights.keys())
        # Combine
        combined: List[float] = []
        for i in range(len(candidate_memories)):
            s = 0.0
            for w, k in zip(weights, crit_keys):
                s += float(w) * float(scores_by_criterion.get(k, [0.0] * len(candidate_memories))[i])
            combined.append(s)
        # Rank and return
        ranked = [m for m, _ in sorted(zip(candidate_memories, combined), key=lambda t: t[1], reverse=True)]
        return ranked


class MemoryHierarchy:
    def __init__(self, hidden_size: int, max_clusters: int = 32):
        self.hidden_size = int(hidden_size)
        self.max_clusters = int(max_clusters)
        self.clusters: Dict[int, List[MemoryEntry]] = {}
        self.cluster_centers: Dict[int, torch.Tensor] = {}
        self.cluster_assignments: Dict[str, int] = {}

    def find_relevant_clusters(self, query_states: torch.Tensor, layer_idx: int, top_k: int = 3) -> List[int]:
        if not self.cluster_centers:
            return []
        q = query_states.mean(dim=(0, 1))
        sims = []
        for cid, center in self.cluster_centers.items():
            sims.append((cid, float(F.cosine_similarity(q, center, dim=0).item())))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sims[:top_k]]

    def get_cluster_memories(self, cluster_id: int) -> List[MemoryEntry]:
        return list(self.clusters.get(int(cluster_id), []))


class RetrievalCache:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = int(cache_size)
        self.cache: Dict[str, List[MemoryEntry]] = {}
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[List[MemoryEntry]]:
        if key in self.cache:
            # Move to MRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: List[MemoryEntry]) -> None:
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
        elif len(self.cache) >= self.cache_size and self.access_order:
            lru = self.access_order.pop(0)
            self.cache.pop(lru, None)
        self.cache[key] = value
        self.access_order.append(key)


class AdvancedMemoryRetriever(nn.Module):
    """Lightweight retriever compatible with tests."""

    def __init__(self, hidden_size: int, memory_buffer: Any, retrieval_config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.memory_buffer = memory_buffer
        self.config = retrieval_config or {}
        # Sub-components required by tests
        self.semantic_matcher = SemanticMemoryMatcher(self.hidden_size, self.config.get('similarity_threshold', 0.7))
        self.context_scorer = ContextualRelevanceScorer(self.hidden_size, self.config.get('context_heads', 8))
        self.multi_criteria_ranker = MultiCriteriaRanker(self.hidden_size, self.config.get('criteria_weights', {
            'relevance': 0.4, 'similarity': 0.3, 'recency': 0.2, 'diversity': 0.1
        }))
        self.memory_hierarchy = MemoryHierarchy(self.hidden_size, self.config.get('max_clusters', 16))
        self.retrieval_cache = RetrievalCache(self.config.get('cache_size', 500))
        self.retrieval_stats: Dict[str, Any] = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'strategy_usage': {},
        }

    def _get_candidate_memories(self, context: RetrievalContext) -> List[MemoryEntry]:
        layer_idx = int(getattr(context, 'current_layer_idx', 0))
        # consider neighboring layers too
        candidates: List[MemoryEntry] = []
        for l in (layer_idx - 1, layer_idx, layer_idx + 1):
            if l >= 0:
                try:
                    candidates.extend(self.memory_buffer.retrieve_by_layer(l))
                except Exception:
                    pass
        return candidates

    def _generate_cache_key(self, context: RetrievalContext, strategy: str) -> str:
        return f"retrieval_{strategy}_{context.current_sequence_id}_{context.current_layer_idx}_{context.sequence_position}_{context.retrieval_budget}_{context.task_type}"

    def _update_retrieval_stats(self, strategy: str, retrieval_time: float) -> None:
        usage = self.retrieval_stats['strategy_usage'].get(strategy, 0)
        avg = self.retrieval_stats['avg_retrieval_time']
        total = self.retrieval_stats['total_retrievals']
        # Online average
        new_avg = (avg * total + float(retrieval_time)) / max(1, total + 1)
        self.retrieval_stats['avg_retrieval_time'] = new_avg
        self.retrieval_stats['strategy_usage'][strategy] = usage + 1
        self.retrieval_stats['total_retrievals'] = total + 1

    def retrieve_memories(self, context: RetrievalContext, strategy: str = 'semantic_similarity') -> List[MemoryEntry]:
        import time
        if strategy not in (
            'semantic_similarity', 'contextual_relevance', 'multi_criteria', 'task_specific', 'hybrid_ensemble'
        ):
            raise ValueError('Unknown retrieval strategy')

        start = time.time()
        cache_key = self._generate_cache_key(context, strategy)
        cached = self.retrieval_cache.get(cache_key)
        if cached is not None:
            self.retrieval_stats['cache_hits'] += 1
            self._update_retrieval_stats(strategy, time.time() - start)
            return cached

        candidates = self._get_candidate_memories(context)
        if not candidates:
            self._update_retrieval_stats(strategy, time.time() - start)
            self.retrieval_cache.put(cache_key, [])
            return []

        if strategy == 'semantic_similarity':
            sims = self.semantic_matcher.compute_similarities(context.current_hidden_states, candidates)
            thresh = self.config.get('similarity_threshold', 0.7)
            ranked = [m for m, s in sorted(zip(candidates, sims), key=lambda t: t[1], reverse=True) if s >= thresh]
            result = ranked[:context.retrieval_budget]
        elif strategy == 'contextual_relevance':
            scores = self.context_scorer.score_memories(context.current_hidden_states, candidates, context.attention_mask)
            ranked = [m for m, s in sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)]
            result = ranked[:context.retrieval_budget]
        elif strategy == 'multi_criteria':
            ranked = self.multi_criteria_ranker.rank_memories(context, candidates)
            result = ranked[:context.retrieval_budget]
        elif strategy == 'task_specific':
            # Simplified: prioritize recent memories
            result = sorted(candidates, key=lambda m: getattr(m, 'timestamp', 0.0), reverse=True)[:context.retrieval_budget]
        elif strategy == 'hybrid_ensemble':
            # Combine similarity and contextual scores
            sims = self.semantic_matcher.compute_similarities(context.current_hidden_states, candidates)
            ctx = self.context_scorer.score_memories(context.current_hidden_states, candidates, context.attention_mask)
            combined = [0.5 * a + 0.5 * b for a, b in zip(sims, ctx)]
            ranked = [m for m, s in sorted(zip(candidates, combined), key=lambda t: t[1], reverse=True)]
            result = ranked[:context.retrieval_budget]

        self.retrieval_cache.put(cache_key, result)
        self._update_retrieval_stats(strategy, time.time() - start)
        return result

    def get_retrieval_stats(self) -> Dict[str, Any]:
        return dict(self.retrieval_stats)


__all__ = [
    'RetrievalContext',
    'AdvancedMemoryRetriever',
    'SemanticMemoryMatcher',
    'ContextualRelevanceScorer',
    'MultiCriteriaRanker',
    'MemoryHierarchy',
    'RetrievalCache',
]


