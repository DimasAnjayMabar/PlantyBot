# evaluation/rag_eval.py
"""
Evaluasi untuk komponen RAG (sampai retrieval + generation)
Metrik: faithfulness, completeness, answer relevance, speed
"""

import json
import os
import re
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
from sentence_transformers import CrossEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

@dataclass
class FaithfulnessMetrics:
    """Metrik faithfulness untuk satu response"""
    score: float
    total_claims: int
    verified_claims: int
    unverified_claims: List[str]
    
@dataclass
class CompletenessMetrics:
    """Metrik completeness untuk satu response"""
    score: float
    causal_elements_found: int
    causal_elements_expected: int
    explanation_depth: int
    missing_elements: List[str]
    
@dataclass
class AnswerRelevanceMetrics:
    """Metrik answer relevance"""
    score: float
    first_relevant_sentence_position: int
    redundancy_ratio: float
    compression_ratio: float


class RAGEvaluator:
    """
    Evaluasi kualitas retrieval dan generation dari RAG pipeline.
    
    Faithfulness: Apakah jawaban hanya berasal dari retrieved chunks?
    Completeness: Apakah hubungan sebab-akibat dijelaskan dengan baik?
    Answer Relevance: Apakah jawaban langsung ke inti pertanyaan?
    """
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.models = rag_pipeline.models
        
        # Load atau buat cross-encoder untuk similarity scoring
        self.similarity_model = CrossEncoder(
            "cross-encoder/stsb-roberta-base",
            device="cpu"  # Ringan, bisa di CPU
        )
    
    def evaluate_faithfulness_batch(
        self,
        responses: List[Dict]  # [{question, answer, chunks, gold_answer?}]
    ) -> List[FaithfulnessMetrics]:
        """
        Evaluasi faithfulness untuk batch responses.
        
        Faithfulness = Proporsi klaim dalam jawaban yang dapat diverifikasi
        dari retrieved chunks.
        """
        results = []
        
        for resp in responses:
            question = resp["question"]
            answer = resp.get("answer", "")
            chunks = resp.get("chunks", [])
            
            if not answer or not chunks:
                results.append(FaithfulnessMetrics(
                    score=0.0, total_claims=0, verified_claims=0, unverified_claims=[]
                ))
                continue
            
            # Extract claims dari answer
            claims = self._extract_claims(answer)
            
            # Verifikasi setiap claim terhadap chunks
            verified_claims = []
            unverified_claims = []
            
            for claim in claims:
                if self._verify_claim_against_chunks(claim, chunks, question):
                    verified_claims.append(claim)
                else:
                    unverified_claims.append(claim)
            
            score = len(verified_claims) / len(claims) if claims else 1.0
            
            results.append(FaithfulnessMetrics(
                score=score,
                total_claims=len(claims),
                verified_claims=len(verified_claims),
                unverified_claims=unverified_claims[:5]  # Simpan max 5
            ))
        
        return results
    
    def evaluate_completeness_batch(
        self,
        responses: List[Dict],
        causal_ground_truth: Optional[Dict] = None
    ) -> List[CompletenessMetrics]:
        """
        Evaluasi completeness untuk batch responses.
        
        Fokus pada penjelasan hubungan sebab-akibat dalam jawaban.
        """
        results = []
        
        for resp in responses:
            question = resp["question"]
            answer = resp.get("answer", "")
            chunks = resp.get("chunks", [])
            
            # Deteksi apakah pertanyaan membutuhkan penjelasan sebab-akibat
            is_causal = self._is_causal_question(question)
            
            if not is_causal:
                # Non-causal questions automatically get full score
                results.append(CompletenessMetrics(
                    score=1.0,
                    causal_elements_found=0,
                    causal_elements_expected=0,
                    explanation_depth=0,
                    missing_elements=[]
                ))
                continue
            
            # Extract causal relations dari answer
            causal_pairs = self._extract_causal_relations(answer)
            
            # Dapatkan expected causal elements dari ground truth atau chunks
            expected_causes = []
            if causal_ground_truth and question in causal_ground_truth:
                expected_causes = causal_ground_truth[question]
            else:
                # Extract dari chunks yang relevan
                expected_causes = self._extract_expected_causes_from_chunks(question, chunks)
            
            # Hitung coverage
            found_elements = [c for c in expected_causes if self._causal_element_in_answer(c, answer)]
            
            coverage = len(found_elements) / len(expected_causes) if expected_causes else 1.0
            
            # Hitung kedalaman penjelasan (berapa layer sebab-akibat)
            depth = self._calculate_causal_depth(answer)
            
            missing = [c for c in expected_causes if c not in found_elements]
            
            results.append(CompletenessMetrics(
                score=coverage,
                causal_elements_found=len(found_elements),
                causal_elements_expected=len(expected_causes),
                explanation_depth=depth,
                missing_elements=missing[:5]
            ))
        
        return results
    
    def evaluate_answer_relevance_batch(
        self,
        responses: List[Dict]
    ) -> List[AnswerRelevanceMetrics]:
        """
        Evaluasi answer relevance untuk batch responses.
        
        Apakah jawaban langsung ke inti atau bertele-tele?
        """
        results = []
        
        for resp in responses:
            question = resp["question"]
            answer = resp.get("answer", "")
            
            if not answer:
                results.append(AnswerRelevanceMetrics(
                    score=0.0, first_relevant_sentence_position=-1,
                    redundancy_ratio=0.0, compression_ratio=0.0
                ))
                continue
            
            # Split menjadi kalimat
            sentences = re.split(r'[.!?]+', answer)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                results.append(AnswerRelevanceMetrics(
                    score=0.0, first_relevant_sentence_position=-1,
                    redundancy_ratio=0.0, compression_ratio=0.0
                ))
                continue
            
            # Hitung relevance tiap kalimat terhadap question
            relevance_scores = self.models.rerank(question, sentences)
            
            # Cari posisi kalimat pertama yang relevan (score > threshold)
            threshold = 0.3  # Bisa disesuaikan
            first_relevant = -1
            for i, score in enumerate(relevance_scores):
                if score > threshold:
                    first_relevant = i
                    break
            
            # Weighted relevance dengan decay berdasarkan posisi
            weights = [0.5, 0.3, 0.15, 0.05]  # Bobot untuk 4 kalimat pertama
            weighted_sum = 0
            for i, (score, weight) in enumerate(zip(relevance_scores[:4], weights[:len(relevance_scores[:4])])):
                weighted_sum += score * weight
            
            relevance_score = weighted_sum / sum(weights[:len(relevance_scores[:4])]) if relevance_scores[:4] else 0
            
            # Hitung redundancy (n-gram repetition)
            tokens = answer.lower().split()
            trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
            if trigrams:
                unique_trigrams = len(set(trigrams))
                redundancy = 1 - (unique_trigrams / len(trigrams)) if trigrams else 0
            else:
                redundancy = 0
            
            # Compression ratio (jawaban vs konten relevan minimal)
            min_expected_length = len(question) * 2  # Asumsi minimal 2x panjang question
            compression = len(answer) / min_expected_length if min_expected_length > 0 else 1
            compression = min(compression, 2.0)  # Cap di 2x
            
            results.append(AnswerRelevanceMetrics(
                score=relevance_score,
                first_relevant_sentence_position=first_relevant,
                redundancy_ratio=redundancy,
                compression_ratio=compression
            ))
        
        return results
    
    def evaluate_speed(
        self,
        test_queries: List[str],
        num_runs: int = 3
    ) -> Dict:
        """
        Evaluasi kecepatan generation (end-to-end).
        
        Returns:
            Dict dengan statistik waktu per komponen
        """
        timing_stats = {
            "total_times": [],
            "retrieval_times": [],
            "enrichment_times": [],
            "rerank_times": [],
            "generation_times": [],
            "memory_times": []
        }
        
        for query in test_queries:
            for _ in range(num_runs):
                # Measure with detailed timing
                t_start = time.perf_counter()
                
                # Retrieval
                t1 = time.perf_counter()
                query_emb = self.models.get_embedding(query)
                candidates = self.pipeline.chroma.retrieve(query_emb, k=12)
                timing_stats["retrieval_times"].append(time.perf_counter() - t1)
                
                # Enrichment
                t2 = time.perf_counter()
                enriched = self.pipeline.neo4j.enrich(candidates, context_window=1)
                timing_stats["enrichment_times"].append(time.perf_counter() - t2)
                
                # Reranking
                t3 = time.perf_counter()
                scores = self.models.rerank(query, [c.context_text for c in enriched])
                timing_stats["rerank_times"].append(time.perf_counter() - t3)
                
                # Generation (simulasi small generation)
                t4 = time.perf_counter()
                # Just a small test generation
                timing_stats["generation_times"].append(0.5)  # Placeholder
                
                timing_stats["total_times"].append(time.perf_counter() - t_start)
        
        return {
            "total": {
                "mean": np.mean(timing_stats["total_times"]),
                "std": np.std(timing_stats["total_times"]),
                "p95": np.percentile(timing_stats["total_times"], 95),
                "p99": np.percentile(timing_stats["total_times"], 99)
            },
            "retrieval": {
                "mean": np.mean(timing_stats["retrieval_times"]),
                "std": np.std(timing_stats["retrieval_times"])
            },
            "enrichment": {
                "mean": np.mean(timing_stats["enrichment_times"]) if timing_stats["enrichment_times"] else 0.0,
                "std": np.std(timing_stats["enrichment_times"]) if timing_stats["enrichment_times"] else 0.0
            },
            "rerank": {
                "mean": np.mean(timing_stats["rerank_times"]),
                "std": np.std(timing_stats["rerank_times"])
            },
            "generation": {
                "mean": np.mean(timing_stats["generation_times"]),
                "std": np.std(timing_stats["generation_times"])
            },
            "total_queries": len(test_queries) * num_runs
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Ekstraksi klaim faktual dari teks jawaban."""
        # Split menjadi kalimat
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            
            # Filter kalimat yang mengandung klaim faktual
            claim_indicators = [
                'adalah', 'merupakan', 'disebabkan oleh', 'menyebabkan',
                'terjadi karena', 'berpengaruh', 'mengakibatkan',
                'is', 'are', 'caused by', 'leads to', 'results in'
            ]
            
            if any(indicator in sent.lower() for indicator in claim_indicators):
                claims.append(sent)
            elif len(sent) > 30:  # Kalimat panjang mungkin mengandung klaim
                claims.append(sent)
        
        return claims[:10]  # Max 10 claims per answer
    
    def _verify_claim_against_chunks(
        self,
        claim: str,
        chunks: List,
        question: str
    ) -> bool:
        """
        Verifikasi apakah claim didukung oleh retrieved chunks.
        
        Menggunakan cross-encoder untuk menghitung similarity.
        """
        if not chunks:
            return False
        
        # Gabungkan semua context dari chunks
        contexts = []
        for chunk in chunks:
            if hasattr(chunk, 'context_text'):
                contexts.append(chunk.context_text)
            elif isinstance(chunk, dict):
                # Raw RAG mengembalikan dict dengan key 'text', fallback ke 'context_text'
                text = chunk.get('text', chunk.get('context_text', ''))
                if text:
                    contexts.append(text)
            elif isinstance(chunk, str):
                contexts.append(chunk)
        
        if not contexts:
            return False
        
        # Hitung similarity claim dengan setiap chunk
        pairs = [[claim, ctx[:1024]] for ctx in contexts]
        scores = self.models.rerank(claim, [ctx[:1024] for ctx in contexts])
        
        # Jika ada chunk dengan similarity > threshold, claim terverifikasi
        threshold = 0.4
        max_score = max(scores) if scores else 0
        
        return max_score >= threshold
    
    def _is_causal_question(self, question: str) -> bool:
        """Deteksi apakah pertanyaan membutuhkan penjelasan sebab-akibat."""
        causal_keywords = [
            'penyebab', 'akibat', 'dampak', 'mengapa', 'kenapa',
            'sebab', 'karena', 'menyebabkan', 'berdampak',
            'cause', 'effect', 'impact', 'why', 'result in',
            'lead to', 'due to', 'consequence'
        ]
        q_lower = question.lower()
        return any(kw in q_lower for kw in causal_keywords)
    
    def _extract_causal_relations(self, text: str) -> List[Tuple[str, str]]:
        """
        Ekstraksi pasangan sebab-akibat dari teks.
        
        Returns:
            List of (cause, effect) tuples
        """
        patterns = [
            (r'(\w+(?:\s+\w+)*)\s+menyebabkan\s+(\w+(?:\s+\w+)*)', 'id'),
            (r'(\w+(?:\s+\w+)*)\s+disebabkan oleh\s+(\w+(?:\s+\w+)*)', 'id'),
            (r'(\w+(?:\s+\w+)*)\s+causes\s+(\w+(?:\s+\w+)*)', 'en'),
            (r'(\w+(?:\s+\w+)*)\s+is caused by\s+(\w+(?:\s+\w+)*)', 'en'),
            (r'karena\s+(\w+(?:\s+\w+)*)\s*,\s*(\w+(?:\s+\w+)*)', 'id'),
            (r'due to\s+(\w+(?:\s+\w+)*)\s*,\s*(\w+(?:\s+\w+)*)', 'en'),
        ]
        
        relations = []
        for pattern, lang in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if len(match) == 2:
                    relations.append((match[0].strip(), match[1].strip()))
        
        return list(set(relations))
    
    def _extract_expected_causes_from_chunks(
        self,
        question: str,
        chunks: List
    ) -> List[str]:
        """Ekstraksi expected causal elements dari retrieved chunks."""
        if not chunks:
            return []
        
        # Gabungkan teks dari chunks
        def _get_chunk_text(c) -> str:
            if hasattr(c, 'context_text'):
                return c.context_text
            if isinstance(c, dict):
                return c.get('text', c.get('context_text', ''))
            if isinstance(c, str):
                return c
            return ''

        all_text = " ".join([_get_chunk_text(c) for c in chunks])[:2000]
        
        # Cari kalimat yang mengandung hubungan sebab-akibat
        causal_sentences = []
        sentences = re.split(r'[.!?]+', all_text)
        
        for sent in sentences:
            if self._is_causal_sentence(sent):
                causal_sentences.append(sent.strip())
        
        # Ekstrak cause phrases
        causes = []
        for sent in causal_sentences:
            cause_match = re.search(r'(\w+(?:\s+\w+)*)\s+(?:menyebabkan|causes|leads to)', sent.lower())
            if cause_match:
                causes.append(cause_match.group(1))
            
            effect_match = re.search(r'(?:disebabkan oleh|is caused by)\s+(\w+(?:\s+\w+)*)', sent.lower())
            if effect_match:
                causes.append(effect_match.group(1))
        
        return list(set(causes))[:5]
    
    def _is_causal_sentence(self, sentence: str) -> bool:
        """Cek apakah kalimat mengandung hubungan sebab-akibat."""
        causal_indicators = [
            'menyebabkan', 'disebabkan', 'karena', 'sehingga',
            'causes', 'caused by', 'due to', 'leads to',
            'result in', 'because', 'therefore'
        ]
        return any(indicator in sentence.lower() for indicator in causal_indicators)
    
    def _causal_element_in_answer(self, element: str, answer: str) -> bool:
        """Cek apakah causal element muncul di answer."""
        return element.lower() in answer.lower()
    
    def _calculate_causal_depth(self, answer: str) -> int:
        """Hitung kedalaman rantai sebab-akibat dalam jawaban."""
        # Cari pola sebab → akibat yang berantai
        causal_chains = re.findall(
            r'(\w+)\s+(?:menyebabkan|causes)\s+(\w+)\s+(?:yang\s+menyebabkan|which\s+causes)\s+(\w+)',
            answer.lower()
        )
        
        if causal_chains:
            return max(len(chain) for chain in causal_chains) + 1
        
        # Single causal pairs
        pairs = self._extract_causal_relations(answer)
        return 1 if pairs else 0
    
    def save_metrics(self, metrics: List, output_path: str):
        """Simpan metrik ke file JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [asdict(m) for m in metrics]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Visualisasi ───────────────────────────────────────────────────────────

    def plot_metrics(
        self,
        faithfulness_metrics: list,
        completeness_metrics: list,
        relevance_metrics: list,
        speed_metrics: dict,
        save_path: str = None
    ):
        """
        Tampilkan bar chart ringkasan untuk semua metrik RAG.

        Layout 2×2:
        - [0,0] Rata-rata skor Faithfulness / Completeness / Relevance
        - [0,1] Distribusi faithfulness (histogram)
        - [1,0] Pipeline component latency
        - [1,1] Completeness — causal depth distribution
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel kiri atas: skor rata-rata ──────────────────────────────────
        ax_scores = fig.add_subplot(gs[0, 0])
        f_mean = float(np.mean([m.score for m in faithfulness_metrics])) if faithfulness_metrics else 0
        c_mean = float(np.mean([m.score for m in completeness_metrics])) if completeness_metrics else 0
        r_mean = float(np.mean([m.score for m in relevance_metrics]))    if relevance_metrics   else 0

        labels = ["Faithfulness", "Completeness", "Relevance"]
        values = [f_mean, c_mean, r_mean]
        colors = ['#3498db', '#e67e22', '#2ecc71']

        bars = ax_scores.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
        ax_scores.set_ylim(0, 1.1)
        ax_scores.set_ylabel("Mean Score")
        ax_scores.set_title("RAG Quality Scores", fontsize=12, weight='bold')
        ax_scores.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax_scores.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', fontsize=10, weight='bold')

        # ── Panel kanan atas: histogram faithfulness ──────────────────────────
        ax_hist = fig.add_subplot(gs[0, 1])
        if faithfulness_metrics:
            f_scores = [m.score for m in faithfulness_metrics]
            ax_hist.hist(f_scores, bins=min(10, len(f_scores)),
                         color='#3498db', edgecolor='black', alpha=0.75)
            ax_hist.axvline(f_mean, color='red', linestyle='--',
                            label=f'Mean: {f_mean:.3f}')
            ax_hist.set_xlabel("Score")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend(fontsize=10)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_title("Faithfulness Distribution", fontsize=12, weight='bold')
        ax_hist.grid(axis='y', alpha=0.3)

        # ── Panel kiri bawah: latency per komponen ────────────────────────────
        ax_speed = fig.add_subplot(gs[1, 0])
        comp_names  = ['Retrieval', 'Rerank', 'Generation', 'Total']
        comp_keys   = ['retrieval', 'rerank', 'generation', 'total']
        comp_times  = [speed_metrics.get(k, {}).get('mean', 0) for k in comp_keys]
        comp_colors = ['#9b59b6', '#1abc9c', '#f39c12', '#e74c3c']

        bars_sp = ax_speed.bar(comp_names, comp_times,
                               color=comp_colors, edgecolor='black', alpha=0.85)
        ax_speed.set_ylabel("Time (seconds)")
        ax_speed.set_title("Pipeline Component Latency", fontsize=12, weight='bold')
        ax_speed.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars_sp, comp_times):
            if val > 0:
                ax_speed.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.005,
                              f'{val:.2f}s', ha='center', fontsize=10, weight='bold')

        # ── Panel kanan bawah: completeness causal depth ──────────────────────
        ax_depth = fig.add_subplot(gs[1, 1])
        if completeness_metrics:
            depths = [m.explanation_depth for m in completeness_metrics]
            depth_counts = {}
            for d in depths:
                depth_counts[d] = depth_counts.get(d, 0) + 1
            dlvls   = sorted(depth_counts.keys())
            dcounts = [depth_counts[d] for d in dlvls]
            ax_depth.bar(dlvls, dcounts, color='#e67e22', edgecolor='black', alpha=0.85)
            ax_depth.set_xticks(dlvls)
            for i, (lvl, cnt) in enumerate(zip(dlvls, dcounts)):
                ax_depth.text(lvl, cnt + 0.1, str(cnt),
                              ha='center', fontsize=10, weight='bold')
        ax_depth.set_xlabel("Causal Depth Level")
        ax_depth.set_ylabel("Number of Responses")
        ax_depth.set_title("Explanation Depth Distribution", fontsize=12, weight='bold')
        ax_depth.grid(axis='y', alpha=0.3)

        plt.suptitle("RAG Pipeline Evaluation — Full Metrics",
                     fontsize=15, weight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[RAGEvaluator] Visualisasi disimpan: {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return fig
    
    def plot_comparison_metrics(
        self,
        graph_metrics: Dict,
        raw_metrics: Dict,
        save_path: str = None
    ):
        """
        Tampilkan bar chart perbandingan Graph RAG vs Raw RAG.
        
        Layout 1×2:
        - RAG Quality Scores (Faithfulness, Completeness, Relevance)
        - Pipeline Component Latency (Retrieval, Rerank, Generation)
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.45, wspace=0.35)

        GREEN = '#2ecc71'
        RED = '#e74c3c'

        # ── Panel 1: RAG Quality Scores ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract scores
        def get_scores(metrics_dict, metric_name):
            if not metrics_dict:
                return 0
            items = metrics_dict.get(metric_name, [])
            if items:
                scores = [item.score for item in items]
                return np.mean(scores)
            return 0
        
        graph_f = get_scores(graph_metrics, "faithfulness")
        graph_c = get_scores(graph_metrics, "completeness")
        graph_r = get_scores(graph_metrics, "relevance")
        
        raw_f = get_scores(raw_metrics, "faithfulness")
        raw_c = get_scores(raw_metrics, "completeness")
        raw_r = get_scores(raw_metrics, "relevance")
        
        # Bar positions
        labels = ['Faithfulness', 'Completeness', 'Answer\nRelevance']
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, [graph_f, graph_c, graph_r], 
                        width, label='Graph RAG', color=GREEN, edgecolor='black', alpha=0.85)
        bars2 = ax1.bar(x + width/2, [raw_f, raw_c, raw_r], 
                        width, label='Raw RAG', color=RED, edgecolor='black', alpha=0.85)
        
        ax1.set_ylabel("Score")
        ax1.set_title("RAG Quality Scores Comparison", fontsize=12, weight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars, prefix in [(bars1, "G"), (bars2, "R")]:
            for bar, val in zip(bars, [graph_f, graph_c, graph_r] if prefix == "G" else [raw_f, raw_c, raw_r]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
        
        # ── Panel 2: Pipeline Component Latency ──────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        
        def get_latency(metrics_dict, component):
            if not metrics_dict:
                return 0
            speed = metrics_dict.get("speed", {})
            return speed.get(component, {}).get("mean", 0)
        
        components = ['retrieval', 'rerank', 'generation']
        comp_labels = ['Retrieval', 'Rerank', 'Generation']
        
        graph_times = [get_latency(graph_metrics, c) for c in components]
        raw_times = [get_latency(raw_metrics, c) for c in components]
        
        x2 = np.arange(len(comp_labels))
        
        bars1 = ax2.bar(x2 - width/2, graph_times, width, 
                        label='Graph RAG', color=GREEN, edgecolor='black', alpha=0.85)
        bars2 = ax2.bar(x2 + width/2, raw_times, width,
                        label='Raw RAG', color=RED, edgecolor='black', alpha=0.85)
        
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Pipeline Component Latency", fontsize=12, weight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(comp_labels)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars, times in [(bars1, graph_times), (bars2, raw_times)]:
            for bar, val in zip(bars, times):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.2f}s', ha='center', fontsize=10)
        
        plt.suptitle("Graph RAG vs Raw RAG Comparison", fontsize=15, weight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[RAGEvaluator] Comparison plot saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return fig