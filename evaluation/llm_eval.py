# evaluation/llm_eval.py
"""
Evaluasi untuk komponen LLM
Metrik: Throughput (dari web official), instruction compliance, answer conciseness
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

@dataclass
class ThroughputMetrics:
    """Metrik throughput LLM dari web official Groq"""
    tokens_per_second: float
    source: str = "groq_official"

@dataclass
class ComplianceMetrics:
    """Metrik kepatuhan instruksi"""
    overall_score: float
    per_instruction: Dict[str, float]

@dataclass
class ConcisenessMetrics:
    """Metrik keketatan jawaban"""
    score: float
    first_relevant_token_position: int
    redundancy_ratio: float
    fluff_ratio: float


class LLMEvaluator:
    """
    Evaluasi performa LLM.
    
    Metrik:
    - Throughput (tokens per second) - dari data official Groq
    - Instruction compliance (kepatuhan terhadap system prompt)
    - Conciseness (langsung ke inti atau bertele-tele)
    """
    
    # Data throughput official Groq untuk Llama 3.3 70B
    # Sumber: https://console.groq.com/docs/models
    OFFICIAL_THROUGHPUT = {
        "qwen3-32b": {"tokens_per_second": 400.0, "source": "https://console.groq.com/docs/models"},
        "gpt-oss-20b": {"tokens_per_second": 1000.0, "source": "https://console.groq.com/docs/models"},
        "llama-3.1-8b-instant": {"tokens_per_second": 560.0, "source": "https://console.groq.com/docs/models"},
        "llama-3.3-70b-versatile": {"tokens_per_second": 280.0, "source": "https://console.groq.com/docs/models"},
        "llama-4-scout-17b-16e": {"tokens_per_second": 750.0, "source": "https://console.groq.com/docs/models"},
        "gpt-oss-120b": {"tokens_per_second": 500.0, "source": "https://console.groq.com/docs/models"},

    }
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.models = rag_pipeline.models
        
    def evaluate_throughput(self) -> Dict:
        """
        Evaluasi throughput menggunakan data official Groq.
        """
        model_name = CONFIG.get("groq_model", "llama-3.3-70b-versatile")
        
        # Hapus prefix provider (contoh: "meta-llama/llama-3.3-70b-versatile" menjadi "llama-3.3-70b-versatile")
        clean_model_name = model_name.split('/')[-1]
        
        # Lakukan pencarian yang lebih fleksibel (partial match)
        matched_key = None
        for key in self.OFFICIAL_THROUGHPUT.keys():
            if key in clean_model_name or clean_model_name in key:
                matched_key = key
                break
                
        throughput_data = self.OFFICIAL_THROUGHPUT.get(
            matched_key if matched_key else clean_model_name, 
            {"tokens_per_second": 100.0, "source": "estimated"}
        )
        
        return {
            "overall_prompts": {
                "mean_tps": throughput_data["tokens_per_second"],
                "std_tps": 0,
                "p50_tps": throughput_data["tokens_per_second"],
                "p95_tps": throughput_data["tokens_per_second"],
                "source": throughput_data["source"],
                "model": model_name
            }
        }
    
    def evaluate_instruction_compliance(
        self,
        test_cases: List[Dict]
    ) -> ComplianceMetrics:
        """
        Evaluasi kepatuhan LLM terhadap system prompt.
        
        Test case format:
        {
            "name": "test_name",
            "system_instruction": "string",
            "user_query": "string",
            "expected_behavior": "bahasa_indonesia|use_saya|no_citation|max_paragraphs|format_list",
            "expected_value": optional
        }
        """
        compliance_scores = {}
        
        for test in test_cases:
            name = test["name"]
            system_instruction = test["system_instruction"]
            user_query = test["user_query"]
            expected = test["expected_behavior"]
            expected_value = test.get("expected_value")
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ]
            
            # Generate response
            response = self.models.groq_client.chat.completions.create(
                model=CONFIG["groq_model"],
                messages=messages,
                max_tokens=256,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
            # Score berdasarkan expected behavior
            score = self._score_compliance(answer, expected, expected_value)
            compliance_scores[name] = score
        
        overall = np.mean(list(compliance_scores.values())) if compliance_scores else 0
        
        return ComplianceMetrics(
            overall_score=overall,
            per_instruction=compliance_scores
        )
    
    def _score_compliance(
        self,
        answer: str,
        expected: str,
        expected_value: any = None
    ) -> float:
        """Skor kepatuhan untuk satu test case."""
        answer_lower = answer.lower()
        
        if expected == "bahasa_indonesia":
            # Deteksi bahasa: hitung kata Indonesia vs Inggris
            id_words = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'adalah'}
            en_words = {'the', 'and', 'of', 'in', 'to', 'for', 'is', 'are'}
            
            id_count = sum(1 for w in answer_lower.split() if w in id_words)
            en_count = sum(1 for w in answer_lower.split() if w in en_words)
            
            if id_count > en_count:
                return 1.0
            elif en_count > id_count:
                return 0.0
            else:
                return 0.5
                
        elif expected == "use_saya":
            # Harus pakai 'saya', jangan pakai 'kami'
            if 'saya' in answer_lower and 'kami' not in answer_lower:
                return 1.0
            elif 'saya' in answer_lower:
                return 0.7
            else:
                return 0.0
                
        elif expected == "no_citation":
            # Jangan sebut sumber
            citation_markers = ['[', ']', 'sumber', 'jurnal', 'penelitian', 'menurut']
            if any(marker in answer_lower for marker in citation_markers):
                return 0.0
            return 1.0
            
        elif expected == "max_paragraphs":
            if expected_value:
                paragraphs = answer.split('\n\n')
                if len(paragraphs) <= expected_value:
                    return 1.0
                else:
                    return max(0, 1 - (len(paragraphs) - expected_value) / expected_value)
            return 0.5
            
        elif expected == "format_list":
            has_bullet = any(marker in answer for marker in ['- ', '* ', '• ', '1.', '2.'])
            return 1.0 if has_bullet else 0.0
            
        return 0.5
    
    def evaluate_conciseness(
        self,
        test_responses: List[Dict]
    ) -> List[ConcisenessMetrics]:
        """
        Evaluasi apakah jawaban langsung ke inti atau bertele-tele.
        """
        results = []
        
        fluff_words = {
            'maaf', 'permisi', 'baiklah', 'jadi', 'sebenarnya',
            'pada dasarnya', 'kurang lebih', 'mungkin', 
            'tentu', 'berikut', 'adalah', 'silakan', 'sebagai', 
            'sorry', 'well', 'actually', 'basically', 'perhaps',
            'maybe', 'just', 'so', 'like', 'you know', 
            'sure', 'certainly', 'here', 'here is' 
        }
        
        for resp in test_responses:
            question = resp["question"]
            answer = resp.get("answer", "")
            
            if not answer:
                results.append(ConcisenessMetrics(
                    score=0.0, first_relevant_token_position=-1,
                    redundancy_ratio=0.0, fluff_ratio=0.0
                ))
                continue
            
            tokens = answer.split()
            if not tokens:
                results.append(ConcisenessMetrics(
                    score=0.0, first_relevant_token_position=-1,
                    redundancy_ratio=0.0, fluff_ratio=0.0
                ))
                continue
            
            # Cari posisi token pertama yang relevan
            question_keywords = set(self._extract_keywords(question))
            first_relevant = -1
            
            for i, token in enumerate(tokens[:30]):
                token_clean = token.lower().strip('.,!?;:')
                if token_clean in question_keywords:
                    first_relevant = i
                    break
            
            if first_relevant == -1 and len(tokens) > 0:
                first_sentence = answer.split('.')[0] if '.' in answer else answer[:200]
                relevance_scores = self.models.rerank(question, [first_sentence])
                if relevance_scores and relevance_scores[0] > 0.3:
                    first_relevant = 0
            
            # Fluff ratio
            fluff_count = sum(1 for token in tokens if token.lower() in fluff_words)
            fluff_ratio = fluff_count / len(tokens)
            
            # Redundancy
            if len(tokens) >= 3:
                trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
                if trigrams:
                    unique_trigrams = len(set(trigrams))
                    redundancy = 1 - (unique_trigrams / len(trigrams))
                else:
                    redundancy = 0
            else:
                redundancy = 0
            
            # Score
            position_score = 1.0 if first_relevant <= 0 else max(0, 1 - (first_relevant / 20))
            conciseness_score = (
                position_score * 0.5 +
                (1 - fluff_ratio) * 0.3 +
                (1 - redundancy) * 0.2
            )
            
            results.append(ConcisenessMetrics(
                score=conciseness_score,
                first_relevant_token_position=first_relevant,
                redundancy_ratio=redundancy,
                fluff_ratio=fluff_ratio
            ))
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        stopwords = {'apa', 'bagaimana', 'mengapa', 'kenapa', 'siapa', 'kapan',
                    'what', 'how', 'why', 'who', 'when', 'which', 'where',
                    'adalah', 'yang', 'dan', 'di', 'ke', 'dari', 'dengan'}
        
        words = text.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w not in stopwords and len(w) > 3]
        
        return list(set(keywords))[:10]
    
    def save_metrics(self, metrics: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── Visualisasi ───────────────────────────────────────────────────────────

    def plot_metrics(
        self,
        throughput_metrics: dict,
        compliance_metrics,
        conciseness_metrics: list,
        save_path: str = None
    ):
        """
        Tampilkan bar chart ringkasan untuk semua metrik LLM.
        
        Layout 1×3:
        - Throughput (TPS)
        - Compliance score per instruksi
        - Conciseness score
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel 1: Throughput ────────────────────────────────────────────────
        ax_tps = fig.add_subplot(gs[0, 0])
        tps_val = (throughput_metrics or {}).get("overall_prompts", {}).get("mean_tps", 0)
        source = (throughput_metrics or {}).get("overall_prompts", {}).get("source", "")
        model = (throughput_metrics or {}).get("overall_prompts", {}).get("model", "")
        
        bar = ax_tps.bar(["Throughput (TPS)"], [tps_val], 
                         color='#3498db', edgecolor='black', alpha=0.85)
        ax_tps.set_ylabel("Tokens per Second")
        ax_tps.set_title(f"LLM Throughput\n{model}", fontsize=11, weight='bold')
        ax_tps.text(0, tps_val + 3, f'{tps_val:.0f} t/s', 
                    ha='center', fontsize=11, weight='bold')
        ax_tps.text(0.5, -0.15, f"Source: {source}", 
                    transform=ax_tps.transAxes, ha='center', fontsize=8)
        ax_tps.grid(axis='y', alpha=0.3)

        # ── Panel 2: Compliance ────────────────────────────────────────────────
        ax_comp = fig.add_subplot(gs[0, 1])
        if compliance_metrics and hasattr(compliance_metrics, 'per_instruction'):
            instr_names = list(compliance_metrics.per_instruction.keys())
            instr_scores = list(compliance_metrics.per_instruction.values())
            short_names = [n[:18] + '…' if len(n) > 18 else n for n in instr_names]

            bars_c = ax_comp.bar(range(len(short_names)), instr_scores,
                                 color='#9b59b6', edgecolor='black', alpha=0.85)
            ax_comp.set_xticks(range(len(short_names)))
            ax_comp.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
            ax_comp.axhline(compliance_metrics.overall_score, color='red',
                            linestyle='--', label=f'Overall: {compliance_metrics.overall_score:.3f}')
            ax_comp.legend(fontsize=9)
            ax_comp.set_ylim(0, 1.15)
            ax_comp.set_ylabel("Compliance Score")
            for bar, val in zip(bars_c, instr_scores):
                ax_comp.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.02,
                             f'{val:.2f}', ha='center', fontsize=9, weight='bold')
        else:
            ax_comp.text(0.5, 0.5, "No compliance data",
                         ha='center', va='center', transform=ax_comp.transAxes)
            ax_comp.axis('off')
        ax_comp.set_title("Instruction Compliance", fontsize=12, weight='bold')
        ax_comp.grid(axis='y', alpha=0.3)

        # ── Panel 3: Conciseness ───────────────────────────────────────────────
        ax_conc = fig.add_subplot(gs[0, 2])
        if conciseness_metrics:
            conc_mean = float(np.mean([m.score for m in conciseness_metrics]))
            fluff_mean = float(np.mean([m.fluff_ratio for m in conciseness_metrics]))
            redun_mean = float(np.mean([m.redundancy_ratio for m in conciseness_metrics]))

            conc_labels = ['Conciseness\nScore', 'Fluff Ratio', 'Redundancy\nRatio']
            conc_vals = [conc_mean, fluff_mean, redun_mean]
            conc_colors = ['#2ecc71', '#e74c3c', '#f39c12']

            bars_conc = ax_conc.bar(conc_labels, conc_vals,
                                    color=conc_colors, edgecolor='black', alpha=0.85)
            ax_conc.set_ylim(0, 1.15)
            ax_conc.set_ylabel("Score / Ratio")
            ax_conc.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars_conc, conc_vals):
                ax_conc.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.02,
                             f'{val:.3f}', ha='center', fontsize=10, weight='bold')
        else:
            ax_conc.text(0.5, 0.5, "No conciseness data",
                         ha='center', va='center', transform=ax_conc.transAxes)
            ax_conc.axis('off')
        ax_conc.set_title("Answer Conciseness Analysis", fontsize=12, weight='bold')

        plt.suptitle("LLM Evaluation — Full Metrics", fontsize=15, weight='bold')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[LLMEvaluator] Visualisasi disimpan: {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return fig
    
    def plot_all_models_metrics(
        self,
        all_models_metrics: Dict,
        save_path: str = None
    ):
        """
        Tampilkan bar chart perbandingan semua model Groq.
        
        Layout 1×3:
        - Tokens per Second (Throughput)
        - Instruction Compliance Score
        - Conciseness Score
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 7))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Extract data per model
        model_names = []
        tps_values = []
        compliance_values = []
        conciseness_values = []
        
        for model_name, metrics in all_models_metrics.items():
            if "error" in metrics:
                continue
            
            # Shorten model names for display
            short_name = model_name.replace("meta-llama/", "").replace("openai/", "").replace("qwen/", "")
            if len(short_name) > 25:
                short_name = short_name[:22] + "..."
            model_names.append(short_name)
            
            # Throughput (from official data)
            throughput = metrics.get("throughput", {})
            tps = throughput.get("overall_prompts", {}).get("mean_tps", 0)
            tps_values.append(tps)
            
            # Compliance
            compliance = metrics.get("compliance")
            comp_score = compliance.overall_score if compliance else 0
            compliance_values.append(comp_score)
            
            # Conciseness
            conciseness = metrics.get("conciseness", [])
            conc_score = np.mean([c.score for c in conciseness]) if conciseness else 0
            conciseness_values.append(conc_score)
        
        if not model_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No model data available", ha='center', va='center')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        # ── Panel 1: Throughput (TPS) ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(model_names)), tps_values, color=colors, edgecolor='black', alpha=0.85)
        ax1.set_ylabel("Tokens per Second (TPS)")
        ax1.set_title("LLM Throughput (Official Data)", fontsize=12, weight='bold')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, tps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
        
        # Add reference line for 100 TPS
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Target 100 TPS')
        ax1.legend(fontsize=8)
        
        # ── Panel 2: Instruction Compliance ──────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(model_names)), compliance_values, color=colors, edgecolor='black', alpha=0.85)
        ax2.set_ylabel("Compliance Score")
        ax2.set_title("Instruction Compliance", fontsize=12, weight='bold')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars2, compliance_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target 0.8')
        ax2.legend(fontsize=8)
        
        # ── Panel 3: Conciseness ─────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(len(model_names)), conciseness_values, color=colors, edgecolor='black', alpha=0.85)
        ax3.set_ylabel("Conciseness Score")
        ax3.set_title("Answer Conciseness", fontsize=12, weight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars3, conciseness_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        ax3.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target 0.7')
        ax3.legend(fontsize=8)
        
        plt.suptitle("LLM Model Comparison - All Groq Models", fontsize=15, weight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"[LLMEvaluator] All models comparison saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)
        return fig