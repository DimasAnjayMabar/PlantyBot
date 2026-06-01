# evaluation/llm_eval.py
"""
Evaluasi untuk komponen LLM
Metrik: throughput (token/s), instruction compliance, answer conciseness
"""

import json
import os
import sys
import time
import numpy as np
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

@dataclass
class ThroughputMetrics:
    """Metrik throughput LLM"""
    tokens_per_second: float
    total_tokens: int
    total_time_s: float
    prompt_tokens: int
    generation_tokens: int
    
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
    fluff_ratio: float  # Proporsi kata tidak informatif


class LLMEvaluator:
    """
    Evaluasi performa LLM secara terpisah dari RAG pipeline.
    
    Metrik:
    - Throughput (tokens per second)
    - Instruction compliance (kepatuhan terhadap system prompt)
    - Conciseness (langsung ke inti atau bertele-tele)
    """
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.models = rag_pipeline.models
        
    def evaluate_throughput(
        self,
        test_prompts: List[str],
        max_tokens_per_response: int = 256,
        num_runs: int = 2
    ) -> Dict:
        """
        Evaluasi token per second untuk berbagai panjang prompt.
        
        Returns:
            Dict dengan metrik throughput per kategori prompt length
        """
        results = {
            "short_prompts": {  # <100 token
                "tps": [],
                "times": [],
                "tokens": []
            },
            "medium_prompts": {  # 100-300 token
                "tps": [],
                "times": [],
                "tokens": []
            },
            "long_prompts": {  # >300 token
                "tps": [],
                "times": [],
                "tokens": []
            },
            "streaming": {
                "tps": [],
                "times": [],
                "tokens": []
            }
        }
        
        # Test non-streaming (batch) mode
        for prompt in test_prompts:
            prompt_tokens = len(prompt) // 4  # Approx token count
            
            # Kategorikan berdasarkan panjang
            if prompt_tokens < 100:
                category = "short_prompts"
            elif prompt_tokens < 300:
                category = "medium_prompts"
            else:
                category = "long_prompts"
            
            for _ in range(num_runs):
                messages = [{"role": "user", "content": prompt}]
                
                start = time.perf_counter()
                tokens_received = 0
                
                # Streaming mode
                stream = self.models.groq_client.chat.completions.create(
                    model=CONFIG["groq_model"],
                    messages=messages,
                    max_tokens=max_tokens_per_response,
                    temperature=0.5,
                    stream=True
                )
                
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        tokens_received += 1
                
                elapsed = time.perf_counter() - start
                tps = tokens_received / elapsed if elapsed > 0 else 0
                
                results[category]["tps"].append(tps)
                results[category]["times"].append(elapsed)
                results[category]["tokens"].append(tokens_received)
                
                # Also record as streaming
                results["streaming"]["tps"].append(tps)
                results["streaming"]["times"].append(elapsed)
                results["streaming"]["tokens"].append(tokens_received)
        
        # Calculate aggregated statistics
        aggregated = {}
        for category, data in results.items():
            if data["tps"]:
                aggregated[category] = {
                    "mean_tps": np.mean(data["tps"]),
                    "std_tps": np.std(data["tps"]),
                    "p50_tps": np.percentile(data["tps"], 50),
                    "p95_tps": np.percentile(data["tps"], 95),
                    "mean_tokens": np.mean(data["tokens"]),
                    "mean_time_s": np.mean(data["times"]),
                    "num_samples": len(data["tps"])
                }
            else:
                aggregated[category] = None
        
        # First token latency (TTFT - Time To First Token)
        aggregated["ttft"] = self._evaluate_ttft(test_prompts[:5])
        
        return aggregated
    
    def _evaluate_ttft(self, test_prompts: List[str]) -> Dict:
        """
        Evaluasi Time To First Token (latency token pertama).
        """
        ttft_values = []
        
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            
            start = time.perf_counter()
            first_token_time = None
            
            stream = self.models.groq_client.chat.completions.create(
                model=CONFIG["groq_model"],
                messages=messages,
                max_tokens=100,
                temperature=0.5,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text and first_token_time is None:
                    first_token_time = time.perf_counter() - start
                    break
            
            if first_token_time:
                ttft_values.append(first_token_time)
        
        return {
            "mean_ttft_ms": np.mean(ttft_values) * 1000 if ttft_values else 0,
            "std_ttft_ms": np.std(ttft_values) * 1000 if ttft_values else 0,
            "p95_ttft_ms": np.percentile(ttft_values, 95) * 1000 if ttft_values else 0
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
            "expected_value": optional (misal 3 untuk max_paragraphs)
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
            
            # Generate response (non-streaming for simplicity)
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
                return 0.7  # Ada 'saya' tapi juga 'kami'
            else:
                return 0.0
                
        elif expected == "no_citation":
            # Jangan sebut sumber
            citation_markers = ['[', ']', 'sumber', 'jurnal', 'penelitian', 'menurut']
            if any(marker in answer_lower for marker in citation_markers):
                return 0.0
            return 1.0
            
        elif expected == "max_paragraphs":
            # Maksimal N paragraf
            if expected_value:
                paragraphs = answer.split('\n\n')
                if len(paragraphs) <= expected_value:
                    return 1.0
                else:
                    return max(0, 1 - (len(paragraphs) - expected_value) / expected_value)
            return 0.5
            
        elif expected == "format_list":
            # Harus dalam format list (bullet points)
            has_bullet = any(marker in answer for marker in ['- ', '* ', '• ', '1.', '2.'])
            return 1.0 if has_bullet else 0.0
            
        elif expected == "no_repetition":
            # Tidak boleh mengulang pertanyaan
            # Cek apakah jawaban mengandung salinan pertanyaan
            # Simplified: cek similarity dengan prompt
            return 0.5  # Placeholder
            
        return 0.5
    
    def evaluate_conciseness(
        self,
        test_responses: List[Dict]  # [{question, answer}]
    ) -> List[ConcisenessMetrics]:
        """
        Evaluasi apakah jawaban langsung ke inti atau bertele-tele.
        
        Metrik:
        - First relevant token position
        - Redundancy (n-gram repetition)
        - Fluff ratio (kata tidak informatif)
        """
        results = []
        
        # Daftar kata tidak informatif (fluff)
        fluff_words = {
            'maaf', 'permisi', 'baiklah', 'jadi', 'sebenarnya',
            'pada dasarnya', 'kurang lebih', 'mungkin',
            'sorry', 'well', 'actually', 'basically', 'perhaps',
            'maybe', 'just', 'so', 'like', 'you know'
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
            
            # Tokenisasi
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
            
            for i, token in enumerate(tokens[:30]):  # Cek 30 token pertama
                token_clean = token.lower().strip('.,!?;:')
                if token_clean in question_keywords or len(token_clean) > 5:
                    first_relevant = i
                    break
            
            # Jika tidak ditemukan, cek similarity dengan question
            if first_relevant == -1 and len(tokens) > 0:
                # Gunakan cross-encoder untuk kalimat pertama
                first_sentence = answer.split('.')[0] if '.' in answer else answer[:200]
                relevance_scores = self.models.rerank(question, [first_sentence])
                if relevance_scores and relevance_scores[0] > 0.3:
                    first_relevant = 0
            
            # Hitung fluff ratio
            fluff_count = sum(1 for token in tokens if token.lower() in fluff_words)
            fluff_ratio = fluff_count / len(tokens)
            
            # Hitung redundancy (trigram repetition)
            if len(tokens) >= 3:
                trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
                if trigrams:
                    unique_trigrams = len(set(trigrams))
                    redundancy = 1 - (unique_trigrams / len(trigrams))
                else:
                    redundancy = 0
            else:
                redundancy = 0
            
            # Score = inverse dari position + fluff + redundancy
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
        """Ekstrak keyword penting dari teks."""
        # Hapus stopwords sederhana
        stopwords = {'apa', 'bagaimana', 'mengapa', 'kenapa', 'siapa', 'kapan',
                    'what', 'how', 'why', 'who', 'when', 'which', 'where',
                    'adalah', 'yang', 'dan', 'di', 'ke', 'dari', 'dengan'}
        
        words = text.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w not in stopwords and len(w) > 3]
        
        return list(set(keywords))[:10]
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Simpan metrik ke file JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)