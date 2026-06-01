# evaluation/report_generator.py
"""
Generator PDF Report dengan Tabel dan Visualisasi Matplotlib
"""

import os
import json
from datetime import datetime
import sys
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from dataclasses import asdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style untuk visualisasi yang lebih professional
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

class ReportGenerator:
    """
    Generate PDF report untuk hasil evaluasi.
    
    Output terpisah per komponen:
    - output/embedder/{timestamp}_embedder_report.pdf
    - output/rag/{timestamp}_rag_report.pdf  
    - output/model/{timestamp}_model_report.pdf
    - output/combined/{timestamp}_full_evaluation.pdf (opsional)
    """
    
    def __init__(self, output_base_dir: str = "output"):
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_embedder_report(
        self,
        retrieval_metrics: Dict,
        graph_vs_raw_metrics: Dict,
        diversity_metrics: Dict,
        config: Dict = None
    ) -> str:
        """
        Generate PDF report untuk evaluasi embedder.
        
        Includes:
        - Retrieval Accuracy @K (tabel + bar chart)
        - MRR per query type (bar chart)
        - Graph vs Raw comparison (side-by-side bar chart)
        - Context diversity metrics (tabel)
        """
        output_dir = os.path.join(self.output_base_dir, "embedder")
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_path = os.path.join(output_dir, f"{self.timestamp}_embedder_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Halaman 1: Cover
            self._add_cover_page(pdf, "Embedder Evaluation Report", config)
            
            # Halaman 2: Retrieval Accuracy Summary
            self._add_retrieval_accuracy_page(pdf, retrieval_metrics)
            
            # Halaman 3: MRR per Query Type
            self._add_mrr_page(pdf, retrieval_metrics)
            
            # Halaman 4: Graph vs Raw Comparison
            self._add_graph_vs_raw_page(pdf, graph_vs_raw_metrics)
            
            # Halaman 5: Context Diversity
            self._add_diversity_page(pdf, diversity_metrics)
            
            # Halaman 6: Detailed Results Table
            self._add_detailed_table_page(pdf, retrieval_metrics, "Embedder Detailed Results")
        
        # Save JSON metrics
        json_path = os.path.join(output_dir, f"{self.timestamp}_embedder_metrics.json")
        self._save_json({
            "retrieval_metrics": retrieval_metrics,
            "graph_vs_raw": graph_vs_raw_metrics,
            "diversity_metrics": diversity_metrics,
            "config": config
        }, json_path)
        
        return pdf_path
    
    def generate_rag_report(
        self,
        faithfulness_metrics: List,
        completeness_metrics: List,
        relevance_metrics: List,
        speed_metrics: Dict,
        config: Dict = None
    ) -> str:
        """
        Generate PDF report untuk evaluasi RAG.
        
        Includes:
        - Faithfulness distribution (histogram + box plot)
        - Completeness score per query type (bar chart)
        - Answer relevance vs faithfulness scatter plot
        - Response time distribution (histogram)
        """
        output_dir = os.path.join(self.output_base_dir, "rag")
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_path = os.path.join(output_dir, f"{self.timestamp}_rag_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Halaman 1: Cover
            self._add_cover_page(pdf, "RAG Pipeline Evaluation Report", config)
            
            # Halaman 2: Faithfulness Analysis
            self._add_faithfulness_page(pdf, faithfulness_metrics)
            
            # Halaman 3: Completeness Analysis
            self._add_completeness_page(pdf, completeness_metrics)
            
            # Halaman 4: Answer Relevance
            self._add_relevance_page(pdf, relevance_metrics)
            
            # Halaman 5: Speed & Performance
            self._add_speed_page(pdf, speed_metrics)
            
            # Halaman 6: Correlation Matrix
            self._add_correlation_page(pdf, faithfulness_metrics, completeness_metrics, relevance_metrics)
        
        # Save JSON metrics
        json_path = os.path.join(output_dir, f"{self.timestamp}_rag_metrics.json")
        self._save_json({
            "faithfulness": [asdict(m) for m in faithfulness_metrics] if faithfulness_metrics else [],
            "completeness": [asdict(m) for m in completeness_metrics] if completeness_metrics else [],
            "relevance": [asdict(m) for m in relevance_metrics] if relevance_metrics else [],
            "speed": speed_metrics,
            "config": config
        }, json_path)
        
        return pdf_path
    
    def generate_model_report(
        self,
        throughput_metrics: Dict,
        compliance_metrics,
        conciseness_metrics: List,
        config: Dict = None
    ) -> str:
        """
        Generate PDF report untuk evaluasi LLM.
        
        Includes:
        - Tokens per second comparison (bar chart)
        - TTFT (Time To First Token) distribution
        - Instruction compliance heatmap
        - Conciseness score distribution
        """
        output_dir = os.path.join(self.output_base_dir, "model")
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_path = os.path.join(output_dir, f"{self.timestamp}_model_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Halaman 1: Cover
            self._add_cover_page(pdf, "LLM Model Evaluation Report", config)
            
            # Halaman 2: Throughput Analysis
            self._add_throughput_page(pdf, throughput_metrics)
            
            # Halaman 3: TTFT Analysis
            self._add_ttft_page(pdf, throughput_metrics)
            
            # Halaman 4: Instruction Compliance
            self._add_compliance_page(pdf, compliance_metrics)
            
            # Halaman 5: Conciseness Analysis
            self._add_conciseness_page(pdf, conciseness_metrics)
        
        # Save JSON metrics
        json_path = os.path.join(output_dir, f"{self.timestamp}_model_metrics.json")
        self._save_json({
            "throughput": throughput_metrics,
            "compliance": asdict(compliance_metrics) if compliance_metrics else {},
            "conciseness": [asdict(m) for m in conciseness_metrics] if conciseness_metrics else [],
            "config": config
        }, json_path)
        
        return pdf_path
    
    def generate_combined_report(
        self,
        all_metrics: Dict,
        config: Dict = None
    ) -> str:
        """
        Generate full combined evaluation report (executive summary).
        """
        output_dir = os.path.join(self.output_base_dir, "combined")
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_path = os.path.join(output_dir, f"{self.timestamp}_full_evaluation.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Halaman 1: Cover
            self._add_cover_page(pdf, "Complete RAG Pipeline Evaluation", config)
            
            # Halaman 2: Executive Summary (tabel ringkasan)
            self._add_executive_summary_page(pdf, all_metrics)
            
            # Halaman 3: Radar Chart (spider plot) untuk overall scores
            self._add_radar_chart_page(pdf, all_metrics)
            
            # Halaman 4: Performance Over Time (jika ada data multiple runs)
            self._add_performance_timeline(pdf, all_metrics)
        
        # Save summary JSON
        json_path = os.path.join(output_dir, f"{self.timestamp}_summary.json")
        self._save_json({
            "summary": self._create_summary_dict(all_metrics),
            "config": config,
            "timestamp": self.timestamp
        }, json_path)
        
        return pdf_path
    
    # ── Private helper methods for page generation ──────────────────────────
    
    def _add_cover_page(self, pdf: PdfPages, title: str, config: Dict = None):
        """Tambahkan halaman cover."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.7, title, fontsize=24, ha='center', weight='bold',
               transform=ax.transAxes)
        ax.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               fontsize=12, ha='center', transform=ax.transAxes)
        
        # Config info
        if config:
            y_pos = 0.45
            ax.text(0.5, y_pos, "Configuration:", fontsize=14, ha='center',
                   weight='bold', transform=ax.transAxes)
            
            config_lines = [
                f"Model: {config.get('groq_model', 'N/A')}",
                f"Embedding: {config.get('embedding_model', 'N/A')}",
                f"Reranker: {config.get('reranker_model', 'N/A')}",
                f"Chroma K: {config.get('chroma_retrieval_k', 'N/A')}",
                f"Reranked K: {config.get('reranked_k', 'N/A')}",
            ]
            
            for i, line in enumerate(config_lines):
                ax.text(0.5, y_pos - 0.05 - (i * 0.04), line,
                       fontsize=10, ha='center', transform=ax.transAxes)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_retrieval_accuracy_page(self, pdf: PdfPages, metrics: Dict):
        """Halaman dengan retrieval accuracy bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        
        # Left: Bar chart Recall@K
        overall = metrics.get("overall", {})
        k_values = [k for k in overall.keys() if k.startswith("recall@")]
        recall_values = [overall[k]["mean"] for k in k_values]
        error_vals = [overall[k]["std"] for k in k_values]
        
        bars = axes[0].bar(k_values, recall_values, yerr=error_vals, capsize=5,
                          color='steelblue', edgecolor='black')
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Retrieval Accuracy @K")
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, recall_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=9)
        
        # Right: Per query type comparison
        per_type = metrics.get("per_query_type", {})
        if per_type:
            query_types = list(per_type.keys())
            recall_at_5 = [per_type[qt].get("recall@5", {}).get("mean", 0) for qt in query_types]
            
            bars = axes[1].bar(query_types, recall_at_5, color='coral', edgecolor='black')
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("Recall@5")
            axes[1].set_title("Recall by Query Type")
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, recall_at_5):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_mrr_page(self, pdf: PdfPages, metrics: Dict):
        """Halaman dengan MRR (Mean Reciprocal Rank) visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract MRR per query type
        per_type = metrics.get("per_query_type", {})
        query_types = list(per_type.keys())
        
        # Calculate MRR from recall metrics (approximation)
        mrr_values = []
        for qt in query_types:
            # Use recall@1 as approximation for MRR
            mrr = per_type[qt].get("recall@1", {}).get("mean", 0)
            mrr_values.append(mrr)
        
        # Also add overall MRR
        overall_mrr = metrics.get("avg_mrr", 0)
        
        # Create grouped bar chart
        x = np.arange(len(query_types))
        width = 0.6
        
        bars = ax.bar(x, mrr_values, width, color='lightgreen', edgecolor='black')
        ax.set_ylabel("MRR / Recall@1")
        ax.set_title("Mean Reciprocal Rank by Query Type")
        ax.set_xticks(x)
        ax.set_xticklabels(query_types, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add overall MRR as horizontal line
        ax.axhline(y=overall_mrr, color='red', linestyle='--', 
                  label=f'Overall MRR: {overall_mrr:.3f}')
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, mrr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_graph_vs_raw_page(self, pdf: PdfPages, metrics: Dict):
        """Halaman perbandingan Graph vs Raw embedder."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # Data
        graph_data = metrics.get("graph", {})
        raw_data = metrics.get("raw", {})
        improvement = metrics.get("improvement", {})
        
        # 1. Context Length
        axes[0].bar(['Graph', 'Raw'], 
                   [graph_data.get('avg_context_length', 0), 
                    raw_data.get('avg_context_length', 0)],
                   color=['#2ecc71', '#e74c3c'])
        axes[0].set_ylabel("Characters")
        axes[0].set_title("Average Context Length")
        axes[0].text(0.5, -0.15, f"Improvement: +{improvement.get('context_length_pct', 0):.1f}%",
                    transform=axes[0].transAxes, ha='center', fontsize=10)
        
        # 2. Entity Coverage
        axes[1].bar(['Graph', 'Raw'],
                   [graph_data.get('avg_entity_coverage', 0),
                    raw_data.get('avg_entity_coverage', 0)],
                   color=['#2ecc71', '#e74c3c'])
        axes[1].set_ylabel("Entities per Chunk")
        axes[1].set_title("Entity Coverage")
        axes[1].text(0.5, -0.15, f"Improvement: +{improvement.get('entity_coverage_pct', 0):.1f}%",
                    transform=axes[1].transAxes, ha='center', fontsize=10)
        
        # 3. Semantic Similarity
        axes[2].bar(['Graph', 'Raw'],
                   [graph_data.get('avg_semantic_similarity', 0),
                    raw_data.get('avg_semantic_similarity', 0)],
                   color=['#2ecc71', '#e74c3c'])
        axes[2].set_ylabel("Similarity Score")
        axes[2].set_title("Query-Context Similarity")
        axes[2].text(0.5, -0.15, f"Improvement: +{improvement.get('semantic_similarity_pct', 0):.1f}%",
                    transform=axes[2].transAxes, ha='center', fontsize=10)
        
        for ax in axes:
            ax.set_ylim(0, max(ax.get_ylim()[1] * 1.1, 1))
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle("Graph Enrichment vs Raw Embedder", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_faithfulness_page(self, pdf: PdfPages, metrics: List):
        """Halaman analisis faithfulness."""
        if not metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        scores = [m.score for m in metrics if hasattr(m, 'score')]
        
        # Left: Histogram
        axes[0].hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(scores):.3f}')
        axes[0].set_xlabel("Faithfulness Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Faithfulness Score Distribution")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        
        # Right: Box plot with categories (if available)
        # Calculate categories based on score ranges
        low = [s for s in scores if s < 0.5]
        medium = [s for s in scores if 0.5 <= s < 0.8]
        high = [s for s in scores if s >= 0.8]
        
        box_data = [low, medium, high]
        box_labels = ['Low (<0.5)', 'Medium (0.5-0.8)', 'High (>0.8)']
        
        bp = axes[1].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#f39c12', '#2ecc71']):
            patch.set_facecolor(color)
        axes[1].set_ylabel("Faithfulness Score")
        axes[1].set_title("Faithfulness by Category")
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add count annotations
        for i, (data, label) in enumerate(zip(box_data, box_labels), 1):
            axes[1].text(i, -0.05, f'n={len(data)}', ha='center', fontsize=9)
        
        plt.suptitle(f"Faithfulness Analysis (n={len(metrics)})", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_completeness_page(self, pdf: PdfPages, metrics: List):
        """Halaman analisis completeness."""
        if not metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        scores = [m.score for m in metrics if hasattr(m, 'score')]
        depths = [m.explanation_depth for m in metrics if hasattr(m, 'explanation_depth')]
        
        # Left: Completeness score distribution
        axes[0].hist(scores, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(scores), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(scores):.3f}')
        axes[0].set_xlabel("Completeness Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Completeness Score Distribution")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        
        # Right: Explanation depth distribution
        depth_counts = {}
        for d in depths:
            depth_counts[d] = depth_counts.get(d, 0) + 1
        
        depth_levels = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depth_levels]
        
        axes[1].bar(depth_levels, counts, color='teal', edgecolor='black')
        axes[1].set_xlabel("Causal Depth Level")
        axes[1].set_ylabel("Number of Responses")
        axes[1].set_title("Explanation Depth Distribution")
        axes[1].set_xticks(depth_levels)
        
        for bar, count in zip(axes[1].patches, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', fontsize=9)
        
        plt.suptitle(f"Completeness Analysis (n={len(metrics)})", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_speed_page(self, pdf: PdfPages, metrics: Dict):
        """Halaman analisis kecepatan."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Component-wise breakdown
        components = ['Retrieval', 'Enrichment', 'Rerank', 'Generation']
        times = [
            metrics.get('retrieval', {}).get('mean', 0),
            metrics.get('enrichment', {}).get('mean', 0) if 'enrichment' in metrics else 0,
            metrics.get('rerank', {}).get('mean', 0),
            metrics.get('generation', {}).get('mean', 0)
        ]
        
        bars = axes[0].bar(components, times, color=['#3498db', '#9b59b6', '#e67e22', '#2ecc71'])
        axes[0].set_ylabel("Time (seconds)")
        axes[0].set_title("Pipeline Component Latency")
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, t in zip(bars, times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{t:.3f}s', ha='center', fontsize=8)
        
        # Right: Total time distribution (if available)
        total_times = metrics.get('total', {}).get('times', [])
        if total_times:
            axes[1].hist(total_times, bins=20, color='cornflowerblue', edgecolor='black', alpha=0.7)
            axes[1].axvline(np.mean(total_times), color='red', linestyle='--',
                           label=f'Mean: {np.mean(total_times):.3f}s')
            axes[1].axvline(np.percentile(total_times, 95), color='orange', linestyle='--',
                           label=f'P95: {np.percentile(total_times, 95):.3f}s')
            axes[1].set_xlabel("Total Response Time (seconds)")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("End-to-End Latency Distribution")
            axes[1].legend()
        
        plt.suptitle("Performance Analysis", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_throughput_page(self, pdf: PdfPages, metrics: Dict):
        """Halaman throughput LLM."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Short', 'Medium', 'Long', 'Overall']
        tps_values = []
        errors = []
        
        for cat in categories:
            cat_key = cat.lower() + "_prompts"
            if cat_key in metrics and metrics[cat_key]:
                tps_values.append(metrics[cat_key].get('mean_tps', 0))
                errors.append(metrics[cat_key].get('std_tps', 0))
            else:
                tps_values.append(0)
                errors.append(0)
        
        bars = ax.bar(categories, tps_values, yerr=errors, capsize=5,
                     color='#1abc9c', edgecolor='black')
        ax.set_ylabel("Tokens per Second")
        ax.set_title("LLM Throughput by Prompt Length")
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, tps_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f} t/s', ha='center', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_compliance_page(self, pdf: PdfPages, compliance_metrics):
        """Halaman kepatuhan instruksi dengan heatmap."""
        if not compliance_metrics or not hasattr(compliance_metrics, 'per_instruction'):
            return
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(compliance_metrics.per_instruction) * 0.4)))
        
        instructions = list(compliance_metrics.per_instruction.keys())
        scores = list(compliance_metrics.per_instruction.values())
        
        # Horizontal bar chart
        y_pos = np.arange(len(instructions))
        bars = ax.barh(y_pos, scores, color='#9b59b6', edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(instructions)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Compliance Score")
        ax.set_title(f"Instruction Compliance (Overall: {compliance_metrics.overall_score:.3f})")
        ax.grid(axis='x', alpha=0.3)
        
        # Color code based on score
        for bar, score in zip(bars, scores):
            if score >= 0.8:
                bar.set_color('#2ecc71')
            elif score >= 0.5:
                bar.set_color('#f39c12')
            else:
                bar.set_color('#e74c3c')
            
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_radar_chart_page(self, pdf: PdfPages, all_metrics: Dict):
        """Halaman radar chart untuk ringkasan overall."""
        # Extract scores dari berbagai metrik
        categories = ['Retrieval\nAccuracy', 'Faithfulness', 'Completeness', 
                     'Answer\nRelevance', 'Speed', 'Instruction\nCompliance']
        
        # Dapatkan scores (normalized ke 0-1)
        scores = []
        
        # Retrieval accuracy (recall@5)
        retrieval = all_metrics.get('embedder', {}).get('retrieval_metrics', {})
        recall_5 = retrieval.get('overall', {}).get('recall@5', {}).get('mean', 0)
        scores.append(recall_5)
        
        # Faithfulness
        faithfulness = all_metrics.get('rag', {}).get('faithfulness', [])
        if faithfulness:
            scores.append(np.mean([f.score for f in faithfulness]))
        else:
            scores.append(0)
        
        # Completeness
        completeness = all_metrics.get('rag', {}).get('completeness', [])
        if completeness:
            scores.append(np.mean([c.score for c in completeness]))
        else:
            scores.append(0)
        
        # Answer relevance
        relevance = all_metrics.get('rag', {}).get('relevance', [])
        if relevance:
            scores.append(np.mean([r.score for r in relevance]))
        else:
            scores.append(0)
        
        # Speed (invers dari latency, normalized)
        speed_metrics = all_metrics.get('rag', {}).get('speed', {})
        total_time = speed_metrics.get('total', {}).get('mean', 1)
        speed_score = max(0, min(1, 1 - (total_time / 5)))  # 5s = 0, 0s = 1
        scores.append(speed_score)
        
        # Instruction compliance
        compliance = all_metrics.get('model', {}).get('compliance', None)
        if compliance:
            scores.append(compliance.overall_score)
        else:
            scores.append(0)
        
        # Radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        scores_loop = scores + scores[:1]
        
        ax.plot(angles, scores_loop, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, scores_loop, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title("Overall Pipeline Performance Radar", fontsize=14, weight='bold', pad=20)
        ax.grid(True)
        
        # Add overall score
        overall_score = np.mean(scores)
        ax.text(0, -0.2, f'Overall Score: {overall_score:.3f}', 
               ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_detailed_table_page(self, pdf: PdfPages, metrics: Dict, title: str):
        """Halaman dengan tabel detail hasil evaluasi."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Prepare table data
        table_data = [["Metric", "Mean", "Std", "Min", "Max"]]
        
        overall = metrics.get("overall", {})
        for k, v in overall.items():
            if isinstance(v, dict) and 'mean' in v:
                table_data.append([
                    k,
                    f"{v.get('mean', 0):.4f}",
                    f"{v.get('std', 0):.4f}",
                    f"{v.get('min', 0):.4f}",
                    f"{v.get('max', 0):.4f}"
                ])
        
        # Add per-query summary
        per_query = metrics.get("per_query", [])
        if per_query:
            table_data.append(["", "", "", "", ""])
            table_data.append(["--- PER QUERY SUMMARY ---", "", "", "", ""])
            
            # Calculate statistics
            recall_5_vals = [q['metrics'].get('recall@5', 0) for q in per_query if 'metrics' in q]
            if recall_5_vals:
                table_data.append([
                    "Recall@5 (distribution)",
                    f"{np.mean(recall_5_vals):.4f}",
                    f"{np.std(recall_5_vals):.4f}",
                    f"{np.min(recall_5_vals):.4f}",
                    f"{np.max(recall_5_vals):.4f}"
                ])
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_executive_summary_page(self, pdf: PdfPages, all_metrics: Dict):
        """Halaman executive summary dengan tabel ringkasan."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Extract key metrics
        summary_data = [
            ["Component", "Metric", "Value", "Status"],
            ["Embedder", "Recall@5", "", ""],
            ["", "Graph Improvement (Context)", "", ""],
            ["", "Graph Improvement (Entities)", "", ""],
            ["", "", "", ""],
            ["RAG", "Faithfulness", "", ""],
            ["", "Completeness", "", ""],
            ["", "Answer Relevance", "", ""],
            ["", "Avg Response Time", "", ""],
            ["", "", "", ""],
            ["LLM", "Throughput (TPS)", "", ""],
            ["", "Instruction Compliance", "", ""],
            ["", "Conciseness", "", ""]
        ]
        
        # Fill values
        # Embedder
        retrieval = all_metrics.get('embedder', {}).get('retrieval_metrics', {})
        recall_5 = retrieval.get('overall', {}).get('recall@5', {}).get('mean', 0)
        summary_data[1][2] = f"{recall_5:.3f}"
        summary_data[1][3] = "✅ Good" if recall_5 > 0.7 else "⚠️ Needs Improvement"
        
        graph_raw = all_metrics.get('embedder', {}).get('graph_vs_raw', {})
        imp_len = graph_raw.get('improvement', {}).get('context_length_pct', 0)
        imp_ent = graph_raw.get('improvement', {}).get('entity_coverage_pct', 0)
        summary_data[2][2] = f"+{imp_len:.1f}%"
        summary_data[2][3] = "✅" if imp_len > 20 else "⚠️"
        summary_data[3][2] = f"+{imp_ent:.1f}%"
        summary_data[3][3] = "✅" if imp_ent > 20 else "⚠️"
        
        # RAG
        rag_data = all_metrics.get('rag', {})
        faithfulness = rag_data.get('faithfulness', [])
        if faithfulness:
            f_score = np.mean([f.score for f in faithfulness])
            summary_data[5][2] = f"{f_score:.3f}"
            summary_data[5][3] = "✅" if f_score > 0.7 else "⚠️"
        
        completeness = rag_data.get('completeness', [])
        if completeness:
            c_score = np.mean([c.score for c in completeness])
            summary_data[6][2] = f"{c_score:.3f}"
            summary_data[6][3] = "✅" if c_score > 0.7 else "⚠️"
        
        relevance = rag_data.get('relevance', [])
        if relevance:
            r_score = np.mean([r.score for r in relevance])
            summary_data[7][2] = f"{r_score:.3f}"
            summary_data[7][3] = "✅" if r_score > 0.6 else "⚠️"
        
        speed = rag_data.get('speed', {})
        total_time = speed.get('total', {}).get('mean', 0)
        summary_data[8][2] = f"{total_time:.2f}s"
        summary_data[8][3] = "✅" if total_time < 3 else "⚠️" if total_time < 5 else "❌"
        
        # LLM
        model_data = all_metrics.get('model', {})
        throughput = model_data.get('throughput', {})
        overall_tps = throughput.get('overall_prompts', {}).get('mean_tps', 0) if throughput else 0
        summary_data[10][2] = f"{overall_tps:.1f} t/s"
        summary_data[10][3] = "✅" if overall_tps > 50 else "⚠️" if overall_tps > 20 else "❌"
        
        compliance = model_data.get('compliance', None)
        if compliance:
            summary_data[11][2] = f"{compliance.overall_score:.3f}"
            summary_data[11][3] = "✅" if compliance.overall_score > 0.8 else "⚠️"
        
        conciseness = model_data.get('conciseness', [])
        if conciseness:
            conc_score = np.mean([c.score for c in conciseness])
            summary_data[12][2] = f"{conc_score:.3f}"
            summary_data[12][3] = "✅" if conc_score > 0.7 else "⚠️"
        
        # Create table
        table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style component rows
        component_rows = [1, 4, 9]
        for row in component_rows:
            if row < len(summary_data):
                for i in range(len(summary_data[0])):
                    table[(row, i)].set_facecolor('#ecf0f1')
                    table[(row, i)].set_text_props(weight='bold')
        
        ax.set_title("Executive Summary", fontsize=16, weight='bold', pad=30)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _create_summary_dict(self, all_metrics: Dict) -> Dict:
        """Buat dictionary ringkasan untuk JSON output."""
        summary = {}
        
        # Embedder summary
        retrieval = all_metrics.get('embedder', {}).get('retrieval_metrics', {})
        summary['embedder'] = {
            'recall@5': retrieval.get('overall', {}).get('recall@5', {}).get('mean', 0),
            'mrr': retrieval.get('avg_mrr', 0),
            'graph_improvement_pct': all_metrics.get('embedder', {})
                .get('graph_vs_raw', {}).get('improvement', {})
                .get('context_length_pct', 0)
        }
        
        # RAG summary
        rag_data = all_metrics.get('rag', {})
        faithfulness = rag_data.get('faithfulness', [])
        completeness = rag_data.get('completeness', [])
        relevance = rag_data.get('relevance', [])
        
        summary['rag'] = {
            'faithfulness': np.mean([f.score for f in faithfulness]) if faithfulness else 0,
            'completeness': np.mean([c.score for c in completeness]) if completeness else 0,
            'answer_relevance': np.mean([r.score for r in relevance]) if relevance else 0,
            'avg_response_time': rag_data.get('speed', {}).get('total', {}).get('mean', 0)
        }
        
        # LLM summary
        model_data = all_metrics.get('model', {})
        throughput = model_data.get('throughput', {})
        compliance = model_data.get('compliance', None)
        conciseness = model_data.get('conciseness', [])
        
        summary['llm'] = {
            'tokens_per_second': throughput.get('overall_prompts', {}).get('mean_tps', 0) if throughput else 0,
            'instruction_compliance': compliance.overall_score if compliance else 0,
            'conciseness': np.mean([c.score for c in conciseness]) if conciseness else 0
        }
        
        summary['overall_score'] = np.mean([
            summary['embedder']['recall@5'],
            summary['rag']['faithfulness'],
            summary['rag']['completeness'],
            summary['rag']['answer_relevance'],
            summary['llm']['tokens_per_second'] / 100,  # Normalize TPS to 0-1 scale
            summary['llm']['instruction_compliance']
        ])
        
        return summary
    
    def _save_json(self, data: Dict, path: str):
        """Simpan data ke file JSON."""
        # Convert non-serializable objects
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert)

    # evaluation/report_generator.py - tambahkan method ini
    def _add_diversity_page(self, pdf: PdfPages, diversity_metrics: Dict):
        """
        Halaman analisis diversitas konteks.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        overall = diversity_metrics.get("overall", {})
        per_query = diversity_metrics.get("per_query", [])
        
        if not overall:
            ax.text(0.5, 0.5, "No diversity data available", 
                ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            return
        
        # Create table data
        table_data = [
            ["Metric", "Value"],
            ["Avg Unique Journals", f"{overall.get('avg_unique_journals', 0):.2f}"],
            ["Avg Unique Headings", f"{overall.get('avg_unique_headings', 0):.2f}"],
            ["Avg Diversity Score", f"{overall.get('avg_diversity_score', 0):.3f}"],
            ["Std Diversity Score", f"{overall.get('std_diversity_score', 0):.3f}"]
        ]
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        ax.set_title("Context Diversity Metrics", fontsize=14, weight='bold', pad=20)
        
        pdf.savefig(fig)
        plt.close(fig)


    def _add_performance_timeline(self, pdf: PdfPages, all_metrics: Dict):
        """
        Halaman performance timeline (placeholder jika tidak ada data multiple runs).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.text(0.5, 0.5, 
                "Performance timeline requires multiple evaluation runs.\n"
                "Run evaluation multiple times to see trends.",
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
        ax.axis('off')
        ax.set_title("Performance Over Time", fontsize=14, weight='bold', pad=20)
        
        pdf.savefig(fig)
        plt.close(fig)


    def _add_relevance_page(self, pdf: PdfPages, relevance_metrics: List):
        """
        Halaman analisis answer relevance.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if not relevance_metrics:
            for ax in axes:
                ax.text(0.5, 0.5, "No relevance data available", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            return
        
        scores = [m.score for m in relevance_metrics if hasattr(m, 'score')]
        first_relevant = [m.first_relevant_sentence_position for m in relevance_metrics 
                        if hasattr(m, 'first_relevant_sentence_position')]
        
        # Left: Relevance score distribution
        axes[0].hist(scores, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(scores):.3f}')
        axes[0].set_xlabel("Relevance Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Answer Relevance Distribution")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        
        # Right: First relevant position (filter positive values)
        valid_positions = [p for p in first_relevant if p >= 0]
        if valid_positions:
            axes[1].hist(valid_positions, bins=min(20, len(set(valid_positions))), 
                        color='#e67e22', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel("Sentence Position")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("First Relevant Sentence Position")
        else:
            axes[1].text(0.5, 0.5, "No relevant sentences detected", 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')
        
        plt.suptitle("Answer Relevance Analysis", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


    def _add_correlation_page(self, pdf: PdfPages, faithfulness_metrics: List,
                            completeness_metrics: List, relevance_metrics: List):
        """
        Halaman korelasi antar metrik.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract scores
        f_scores = [m.score for m in faithfulness_metrics] if faithfulness_metrics else []
        c_scores = [m.score for m in completeness_metrics] if completeness_metrics else []
        r_scores = [m.score for m in relevance_metrics] if relevance_metrics else []
        
        if not (f_scores and c_scores and r_scores):
            ax.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            return
        
        # Create correlation matrix
        import pandas as pd
        min_len = min(len(f_scores), len(c_scores), len(r_scores))
        data = pd.DataFrame({
            'Faithfulness': f_scores[:min_len],
            'Completeness': c_scores[:min_len],
            'Answer Relevance': r_scores[:min_len]
        })
        
        corr = data.corr()
        
        # Plot heatmap
        im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")
        
        ax.set_title("Correlation Between Metrics", fontsize=14, weight='bold')
        
        pdf.savefig(fig)
        plt.close(fig)


    def _add_conciseness_page(self, pdf: PdfPages, conciseness_metrics: List):
        """
        Halaman analisis conciseness untuk LLM evaluation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if not conciseness_metrics:
            for ax in axes:
                ax.text(0.5, 0.5, "No conciseness data available", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            return
        
        scores = [m.score for m in conciseness_metrics if hasattr(m, 'score')]
        fluff_ratios = [m.fluff_ratio for m in conciseness_metrics if hasattr(m, 'fluff_ratio')]
        
        # Left: Conciseness score distribution
        axes[0].hist(scores, bins=20, color='#1abc9c', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(scores):.3f}')
        axes[0].set_xlabel("Conciseness Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Conciseness Distribution")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        
        # Right: Fluff ratio distribution
        axes[1].hist(fluff_ratios, bins=20, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(fluff_ratios), color='blue', linestyle='--', 
                    label=f'Mean: {np.mean(fluff_ratios):.3f}')
        axes[1].set_xlabel("Fluff Ratio")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Uninformative Words Ratio")
        axes[1].legend()
        
        plt.suptitle("Answer Conciseness Analysis", fontsize=14, weight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


    def _add_ttft_page(self, pdf: PdfPages, throughput_metrics: Dict):
        """
        Halaman Time-to-First-Token analysis.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ttft_data = throughput_metrics.get("ttft", {})
        
        if not ttft_data:
            ax.text(0.5, 0.5, "No TTFT data available", 
                ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
            return
        
        metrics = ['mean_ttft_ms', 'p95_ttft_ms']
        values = [ttft_data.get(m, 0) for m in metrics]
        
        bars = ax.bar(metrics, values, color='#9b59b6', edgecolor='black')
        ax.set_ylabel("Milliseconds (ms)")
        ax.set_title("Time to First Token (TTFT)")
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f} ms', ha='center', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)