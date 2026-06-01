# evaluation/embedder_eval.py
"""
Evaluasi untuk komponen Embedder
Metrik: Graph vs Raw comparison only
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


class EmbedderEvaluator:
    """
    Evaluasi performa embedding model - Graph vs Raw comparison.
    
    Memerlukan test dataset dengan format:
    {
        "queries": [
            {
                "query": "string",
                "expected_chunk_ids": ["id1", "id2"],
                "query_type": "factual|causal|comparative",
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
            raw_collection = self.chroma.client.get_collection(
                CONFIG["raw_collection"]
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
    
    def _count_entities(self, text: str) -> int:
        """Hitung jumlah entitas (NER) dalam teks."""
        try:
            # Gunakan NER pipeline yang sudah ada
            entities = self.models.nlp_en_pipeline(text[:512])
            # Filter by score threshold
            high_confidence = [e for e in entities if e.get("score", 0) >= 0.7]
            return len(high_confidence)
        except Exception:
            # Fallback: hitung proper nouns
            import re
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return len(set(proper_nouns))
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Simpan metrik ke file JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Visualisasi ───────────────────────────────────────────────────────────

    def plot_metrics(
        self,
        graph_vs_raw_metrics: Dict,
        save_path: str = None
    ):
        """
        Tampilkan bar chart untuk graph vs raw metrics.
        
        3 panel:
        - Context length
        - Entity coverage  
        - Query-Context similarity
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.45, wspace=0.35)

        GREEN = '#2ecc71'
        RED   = '#e74c3c'

        g = graph_vs_raw_metrics.get("graph", {})
        r = graph_vs_raw_metrics.get("raw",   {})
        imp = graph_vs_raw_metrics.get("improvement", {})

        panels = [
            ("avg_context_length",  "Characters",         "Average Context Length",   gs[0, 0]),
            ("avg_entity_coverage", "Entities per Chunk", "Entity Coverage",          gs[0, 1]),
            ("avg_semantic_similarity", "Similarity Score", "Query-Context Similarity", gs[0, 2]),
        ]
        imp_keys = ["context_length_pct", "entity_coverage_pct", "semantic_similarity_pct"]

        for (metric_key, ylabel, title, subplot_spec), imp_key in zip(panels, imp_keys):
            ax = fig.add_subplot(subplot_spec)
            vals = [g.get(metric_key, 0), r.get(metric_key, 0)]
            bars_ = ax.bar(['Graph', 'Raw'], vals,
                           color=[GREEN, RED], edgecolor='black', alpha=0.85)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=11, weight='bold')
            ax.set_ylim(0, max(max(vals) * 1.2, 1))
            ax.grid(axis='y', alpha=0.3)
            for bar_, val_ in zip(bars_, vals):
                ax.text(bar_.get_x() + bar_.get_width() / 2,
                        bar_.get_height() + max(vals) * 0.03,
                        f'{val_:.2f}', ha='center', fontsize=9, weight='bold')
            pct = imp.get(imp_key, 0)
            ax.text(0.5, -0.18,
                    f"Improvement: +{pct:.1f}%",
                    transform=ax.transAxes, ha='center',
                    fontsize=9, color='darkgreen', weight='bold')

        fig.suptitle("Graph Enrichment vs Raw Embedder", fontsize=15, weight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[EmbedderEvaluator] Visualisasi disimpan: {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return fig