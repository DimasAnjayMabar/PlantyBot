# evaluation/embedder_eval.py
"""
Evaluasi untuk komponen Embedder
Metrik: retrieval accuracy, context relevance (graph vs raw)
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

@dataclass
class RetrievalMetrics:
    """Metrik retrieval untuk satu query"""
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_5: float  # Normalized Discounted Cumulative Gain
    query_type: str  # 'factual', 'causal', 'comparative'
    
@dataclass
class ContextComparison:
    """Perbandingan graph vs raw embedder"""
    graph_avg_context_length: float
    raw_avg_context_length: float
    graph_avg_entities: float
    raw_avg_entities: float
    graph_avg_semantic_similarity: float
    raw_avg_semantic_similarity: float
    improvement_pct_length: float
    improvement_pct_entities: float


class EmbedderEvaluator:
    """
    Evaluasi performa embedding model dan retrieval.
    
    Memerlukan test dataset dengan format:
    {
        "queries": [
            {
                "query": "string",
                "expected_chunk_ids": ["id1", "id2"],  # relevant chunks
                "query_type": "factual|causal|comparative",
                "relevance_scores": {"chunk_id": score}  # optional graded relevance
            }
        ]
    }
    """
    
    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: Instance RAGPipeline yang sudah running
        """
        self.pipeline = rag_pipeline
        self.models = rag_pipeline.models
        self.chroma = rag_pipeline.chroma
        self.neo4j = rag_pipeline.neo4j
        
    def evaluate_retrieval_accuracy(
        self,
        test_queries: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Hitung Recall@K, MRR, dan nDCG@K untuk semua test queries.
        
        Returns:
            Dict dengan metrik agregat dan per-query-type
        """
        results = {
            "overall": {
                f"recall@{k}": [] for k in k_values
            },
            "per_query_type": defaultdict(lambda: {
                f"recall@{k}": [] for k in k_values
            }),
            "per_query": []
        }
        
        mrr_scores = []
        
        for query_data in test_queries:
            query = query_data["query"]
            expected_ids = set(query_data["expected_chunk_ids"])
            query_type = query_data.get("query_type", "unknown")
            graded_relevance = query_data.get("relevance_scores", {})
            
            # Get retrieval results
            query_emb = self.models.get_embedding(query)
            candidates = self.chroma.retrieve(query_emb, k=max(k_values))
            retrieved_ids = [c.isi_id for c in candidates]
            
            # Calculate metrics for this query
            query_metrics = {}
            
            # Recall@K and nDCG@K
            for k in k_values:
                retrieved_k = retrieved_ids[:k]
                relevant_k = [rid for rid in retrieved_k if rid in expected_ids]
                
                # Recall
                recall = len(relevant_k) / len(expected_ids) if expected_ids else 0
                results["overall"][f"recall@{k}"].append(recall)
                results["per_query_type"][query_type][f"recall@{k}"].append(recall)
                query_metrics[f"recall@{k}"] = recall
                
                # nDCG@K (if graded relevance available)
                if graded_relevance:
                    dcg = 0
                    idcg = 0
                    for i, rid in enumerate(retrieved_k[:k]):
                        rel = graded_relevance.get(rid, 0)
                        dcg += rel / np.log2(i + 2)
                    
                    # Ideal DCG: sort relevance scores descending
                    sorted_scores = sorted(graded_relevance.values(), reverse=True)[:k]
                    for i, rel in enumerate(sorted_scores):
                        idcg += rel / np.log2(i + 2)
                    
                    ndcg = dcg / idcg if idcg > 0 else 0
                    query_metrics[f"ndcg@{k}"] = ndcg
            
            # MRR (Mean Reciprocal Rank)
            rank = None
            for i, rid in enumerate(retrieved_ids):
                if rid in expected_ids:
                    rank = i + 1
                    break
            mrr = 1 / rank if rank else 0
            mrr_scores.append(mrr)
            query_metrics["mrr"] = mrr
            
            # Store per-query results
            results["per_query"].append({
                "query": query[:100],
                "query_type": query_type,
                "metrics": query_metrics,
                "retrieved_count": len(retrieved_ids),
                "expected_count": len(expected_ids)
            })
        
        # Aggregate results
        final_results = {
            "overall": {},
            "per_query_type": {},
            "per_query": results["per_query"],
            "avg_mrr": np.mean(mrr_scores),
            "std_mrr": np.std(mrr_scores),
            "total_queries": len(test_queries)
        }
        
        for k in k_values:
            final_results["overall"][f"recall@{k}"] = {
                "mean": np.mean(results["overall"][f"recall@{k}"]),
                "std": np.std(results["overall"][f"recall@{k}"]),
                "min": np.min(results["overall"][f"recall@{k}"]),
                "max": np.max(results["overall"][f"recall@{k}"])
            }
        
        for query_type, metrics in results["per_query_type"].items():
            final_results["per_query_type"][query_type] = {}
            for k in k_values:
                vals = metrics[f"recall@{k}"]
                if vals:
                    final_results["per_query_type"][query_type][f"recall@{k}"] = {
                        "mean": np.mean(vals),
                        "std": np.std(vals),
                        "count": len(vals)
                    }
        
        return final_results
    
    def compare_graph_vs_raw(
        self,
        test_queries: List[Dict],
        k: int = 5
    ) -> Dict:
        """
        Bandingkan context enrichment antara Graph (improved) dan Raw embedder.
        
        Metrik:
        - Context length (characters)
        - Entity coverage (jumlah entitas unik per chunk)
        - Semantic similarity gain (perbedaan similarity setelah enrichment)
        """
        results = {
            "graph": {
                "context_lengths": [],
                "entity_counts": [],
                "similarity_to_query": []
            },
            "raw": {
                "context_lengths": [],
                "entity_counts": [],
                "similarity_to_query": []
            },
            "per_query": []
        }
        
        for query_data in test_queries:
            query = query_data["query"]
            query_emb = self.models.get_embedding(query)
            
            # ── Graph pipeline (improved) ─────────────────────────────────────
            candidates = self.chroma.retrieve(query_emb, k=k)
            enriched = self.neo4j.enrich(candidates, context_window=1)
            
            for chunk in enriched:
                context_text = chunk.context_text
                results["graph"]["context_lengths"].append(len(context_text))
                results["graph"]["entity_counts"].append(
                    self._count_entities(context_text)
                )
                
                # Semantic similarity between query and context
                ctx_emb = self.models.get_embedding(context_text)
                similarity = 1 - np.dot(query_emb, ctx_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(ctx_emb) + 1e-8
                )
                results["graph"]["similarity_to_query"].append(similarity)
            
            # ── Raw pipeline (regular) ────────────────────────────────────────
            # ✅ BENAR
            raw_collection = self.chroma.client.get_collection(
                CONFIG["raw_collection"]  # ← langsung dari CONFIG
            )
            raw_results = raw_collection.query(
                query_embeddings=[query_emb],
                n_results=k,
                include=["documents", "metadatas"]
            )
            
            for i, doc in enumerate(raw_results["documents"][0]):
                raw_text = doc
                results["raw"]["context_lengths"].append(len(raw_text))
                results["raw"]["entity_counts"].append(
                    self._count_entities(raw_text)
                )
                
                raw_emb = self.models.get_embedding(raw_text)
                similarity = 1 - np.dot(query_emb, raw_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(raw_emb) + 1e-8
                )
                results["raw"]["similarity_to_query"].append(similarity)
            
            results["per_query"].append({
                "query": query[:100],
                "graph_avg_length": np.mean([len(c.context_text) for c in enriched]) if enriched else 0,
                "raw_avg_length": np.mean(results["raw"]["context_lengths"][-k:]) if raw_results["documents"][0] else 0,
                "graph_avg_entities": np.mean([self._count_entities(c.context_text) for c in enriched]) if enriched else 0,
                "raw_avg_entities": np.mean([self._count_entities(d) for d in raw_results["documents"][0]]) if raw_results["documents"][0] else 0
            })
        
        # Calculate statistics
        graph_mean_len = np.mean(results["graph"]["context_lengths"]) if results["graph"]["context_lengths"] else 0
        raw_mean_len = np.mean(results["raw"]["context_lengths"]) if results["raw"]["context_lengths"] else 0
        
        graph_mean_ent = np.mean(results["graph"]["entity_counts"]) if results["graph"]["entity_counts"] else 0
        raw_mean_ent = np.mean(results["raw"]["entity_counts"]) if results["raw"]["entity_counts"] else 0
        
        graph_mean_sim = np.mean(results["graph"]["similarity_to_query"]) if results["graph"]["similarity_to_query"] else 0
        raw_mean_sim = np.mean(results["raw"]["similarity_to_query"]) if results["raw"]["similarity_to_query"] else 0
        
        return {
            "graph": {
                "avg_context_length": graph_mean_len,
                "std_context_length": np.std(results["graph"]["context_lengths"]) if results["graph"]["context_lengths"] else 0,
                "avg_entity_coverage": graph_mean_ent,
                "std_entity_coverage": np.std(results["graph"]["entity_counts"]) if results["graph"]["entity_counts"] else 0,
                "avg_semantic_similarity": graph_mean_sim
            },
            "raw": {
                "avg_context_length": raw_mean_len,
                "std_context_length": np.std(results["raw"]["context_lengths"]) if results["raw"]["context_lengths"] else 0,
                "avg_entity_coverage": raw_mean_ent,
                "std_entity_coverage": np.std(results["raw"]["entity_counts"]) if results["raw"]["entity_counts"] else 0,
                "avg_semantic_similarity": raw_mean_sim
            },
            "improvement": {
                "context_length_pct": ((graph_mean_len - raw_mean_len) / raw_mean_len * 100) if raw_mean_len > 0 else 0,
                "entity_coverage_pct": ((graph_mean_ent - raw_mean_ent) / raw_mean_ent * 100) if raw_mean_ent > 0 else 0,
                "semantic_similarity_pct": ((graph_mean_sim - raw_mean_sim) / raw_mean_sim * 100) if raw_mean_sim > 0 else 0
            },
            "per_query": results["per_query"]
        }
    
    def evaluate_context_diversity(
        self,
        test_queries: List[Dict],
        k: int = 5
    ) -> Dict:
        """
        Evaluasi diversitas konteks yang direturn.
        
        Metrik:
        - Unique jurnal count per query
        - Unique sub_judul count per query
        - Information overlap antar chunks
        """
        results = []
        
        for query_data in test_queries:
            query = query_data["query"]
            query_emb = self.models.get_embedding(query)
            
            candidates = self.chroma.retrieve(query_emb, k=k)
            enriched = self.neo4j.enrich(candidates, context_window=1)
            
            unique_journals = set(c.jurnal_id for c in enriched)
            unique_headings = set(c.sub_judul for c in enriched)
            
            # Calculate pairwise cosine similarity between chunks
            chunk_texts = [c.context_text for c in enriched]
            if len(chunk_texts) > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                
                chunk_embs = self.models.embed_batch_safe(chunk_texts)
                similarity_matrix = cosine_similarity(chunk_embs)
                # Exclude diagonal
                mask = np.ones(similarity_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                avg_similarity = similarity_matrix[mask].mean() if mask.any() else 0
                diversity_score = 1 - avg_similarity
            else:
                diversity_score = 1.0
            
            results.append({
                "query": query[:100],
                "unique_journals": len(unique_journals),
                "unique_headings": len(unique_headings),
                "diversity_score": diversity_score,
                "total_chunks": len(enriched)
            })
        
        return {
            "overall": {
                "avg_unique_journals": np.mean([r["unique_journals"] for r in results]),
                "avg_unique_headings": np.mean([r["unique_headings"] for r in results]),
                "avg_diversity_score": np.mean([r["diversity_score"] for r in results]),
                "std_diversity_score": np.std([r["diversity_score"] for r in results])
            },
            "per_query": results
        }
    
    def _count_entities(self, text: str) -> int:
        """Hitung jumlah entitas (NER) dalam teks."""
        try:
            # Gunakan NER pipeline yang sudah ada
            entities = self.models.nlp_en_pipeline(text[:512])  # Batasi panjang
            # Filter by score threshold
            high_confidence = [e for e in entities if e.get("score", 0) >= 0.7]
            return len(high_confidence)
        except Exception:
            # Fallback: hitung proper nouns (kata dengan huruf kapital di awal)
            import re
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return len(set(proper_nouns))
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Simpan metrik ke file JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)