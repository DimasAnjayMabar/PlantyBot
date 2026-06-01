# evaluation/main.py - VERSI REVISI

"""
Main evaluator yang mengintegrasikan semua komponen.
Output: PNG plots + JSON metrics (no PDF)
"""

import os
import json
import logging
import sys
import time
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

from .embedder_eval import EmbedderEvaluator
from .rag_eval import RAGEvaluator
from .llm_eval import LLMEvaluator

from config import CONFIG, GROQ_ALLOWED_MODELS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ── Rate-limit helper ─────────────────────────────────────────────────────────

def _parse_retry_after(error_message: str) -> float:
    import re
    match = re.search(r'(?:(\d+)m)?(\d+(?:\.\d+)?)s', error_message)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2) or 0)
        return minutes * 60 + seconds
    return 60.0


def _call_with_rate_limit_retry(func, *args, max_retries: int = 3, **kwargs):
    try:
        from groq import RateLimitError
    except ImportError:
        return func(*args, **kwargs)

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            wait_sec = _parse_retry_after(str(e))
            if attempt < max_retries - 1:
                logger.warning(
                    f"Groq rate limit hit (attempt {attempt + 1}/{max_retries}). "
                    f"Menunggu {wait_sec:.0f}s sebelum retry..."
                )
                time.sleep(wait_sec + 5)
            else:
                logger.error(f"Rate limit masih hit setelah {max_retries} percobaan.")
                raise
        except Exception:
            raise


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGEvaluationPipeline:
    """
    Main orchestrator untuk menjalankan evaluasi lengkap.
    Output: PNG plots + JSON metrics
    """
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.embedder_eval = EmbedderEvaluator(rag_pipeline)
        self.rag_eval = RAGEvaluator(rag_pipeline)
        self.llm_eval = LLMEvaluator(rag_pipeline)
        
        self.test_dataset = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_test_dataset(self, dataset_path: str) -> Dict:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.test_dataset = json.load(f)
        logger.info(f"Loaded test dataset from {dataset_path}")
        return self.test_dataset
    
    def evaluate_embedder(self, queries: List[Dict] = None) -> Dict:
        """Evaluasi komponen embedder - Graph vs Raw."""
        if queries is None:
            queries = self.test_dataset.get("embedder_queries", [])
        
        if not queries:
            logger.warning("No embedder test queries found")
            return {}
        
        logger.info(f"Running embedder evaluation on {len(queries)} queries...")
        
        graph_vs_raw = self.embedder_eval.compare_graph_vs_raw(queries)
        
        return {
            "graph_vs_raw": graph_vs_raw
        }
    
    def evaluate_rag_both_pipelines(
        self, 
        queries: List[Dict] = None
    ) -> Dict:
        """
        Evaluasi kedua pipeline RAG:
        - Graph RAG (improved dengan Neo4j enrichment)
        - Raw RAG (regular tanpa enrichment)
        
        Returns perbandingan metrics.
        """
        if queries is None:
            queries = self.test_dataset.get("rag_queries", [])
        
        if not queries:
            logger.warning("No RAG test queries found")
            return {}
        
        logger.info(f"Running RAG evaluation on {len(queries)} queries...")
        
        # ── Graph RAG (improved) ──────────────────────────────────────────────
        logger.info("  Evaluating GRAPH RAG pipeline...")
        graph_responses = self._evaluate_pipeline_responses(
            queries, use_raw=False
        )
        
        # ── Raw RAG (regular) ────────────────────────────────────────────────
        logger.info("  Evaluating RAW RAG pipeline...")
        # Temporarily switch rag_mode to 'regular'
        original_mode = CONFIG.get("rag_mode", "improved")
        CONFIG["rag_mode"] = "regular"
        
        raw_responses = self._evaluate_pipeline_responses(
            queries, use_raw=True
        )
        
        # Restore original mode
        CONFIG["rag_mode"] = original_mode
        
        if not graph_responses and not raw_responses:
            logger.error("Both pipelines failed.")
            return {}
        
        # Calculate metrics for both
        results = {
            "graph_rag": {},
            "raw_rag": {}
        }
        
        if graph_responses:
            faithfulness = self.rag_eval.evaluate_faithfulness_batch(graph_responses)
            completeness = self.rag_eval.evaluate_completeness_batch(
                graph_responses, 
                self.test_dataset.get("causal_ground_truth", {})
            )
            relevance = self.rag_eval.evaluate_answer_relevance_batch(graph_responses)
            speed = self.rag_eval.evaluate_speed([q["question"] for q in queries if q.get("question")])
            
            results["graph_rag"] = {
                "faithfulness": faithfulness,
                "completeness": completeness,
                "relevance": relevance,
                "speed": speed,
                "responses": graph_responses
            }
        
        if raw_responses:
            faithfulness = self.rag_eval.evaluate_faithfulness_batch(raw_responses)
            completeness = self.rag_eval.evaluate_completeness_batch(
                raw_responses, 
                self.test_dataset.get("causal_ground_truth", {})
            )
            relevance = self.rag_eval.evaluate_answer_relevance_batch(raw_responses)
            speed = self.rag_eval.evaluate_speed([q["question"] for q in queries if q.get("question")])
            
            results["raw_rag"] = {
                "faithfulness": faithfulness,
                "completeness": completeness,
                "relevance": relevance,
                "speed": speed,
                "responses": raw_responses
            }
        
        return results
    
    def _evaluate_pipeline_responses(
        self, 
        queries: List[Dict], 
        use_raw: bool = False
    ) -> List[Dict]:
        """
        Eksekusi pipeline untuk list queries.
        """
        responses = []
        skipped = 0
        
        for idx, q in enumerate(queries):
            question = q.get("question", "")
            if not question:
                continue
            
            logger.info(f"  Query {idx + 1}/{len(queries)}: {question[:60]}...")
            
            def _run_query():
                if use_raw:
                    # Raw RAG: set rag_mode ke 'regular'
                    original = CONFIG.get("rag_mode", "improved")
                    CONFIG["rag_mode"] = "regular"
                    try:
                        return self.pipeline.process_query(
                            question,
                            chat_id=q.get("chat_id"),
                            user_id=q.get("user_id")
                        )
                    finally:
                        CONFIG["rag_mode"] = original
                else:
                    return self.pipeline.process_query(
                        question,
                        chat_id=q.get("chat_id"),
                        user_id=q.get("user_id")
                    )
            
            try:
                response = _call_with_rate_limit_retry(_run_query, max_retries=3)
            except Exception as e:
                logger.error(f"  Query dilewati karena error: {e}")
                skipped += 1
                continue
            
            answer_text = ""
            try:
                if hasattr(response.answer, '__iter__') and not isinstance(response.answer, str):
                    for chunk in response.answer:
                        answer_text += chunk
                else:
                    answer_text = str(response.answer)
            except Exception as e:
                from groq import RateLimitError
                if isinstance(e, RateLimitError):
                    wait_sec = _parse_retry_after(str(e))
                    logger.warning(f"  Rate limit saat streaming. Menunggu {wait_sec:.0f}s...")
                    time.sleep(wait_sec + 5)
                    try:
                        response = _run_query()
                        if hasattr(response.answer, '__iter__') and not isinstance(response.answer, str):
                            for chunk in response.answer:
                                answer_text += chunk
                        else:
                            answer_text = str(response.answer)
                    except Exception as e2:
                        logger.error(f"  Retry stream gagal: {e2}")
                        skipped += 1
                        continue
                else:
                    logger.error(f"  Error membaca stream: {e}")
                    skipped += 1
                    continue
            
            responses.append({
                "question": question,
                "answer": answer_text,
                "chunks": response.final_chunks,
                "processing_time": response.processing_time,
                "gold_answer": q.get("gold_answer"),
                "expected_causal": q.get("expected_causal_relations", [])
            })
        
        if skipped > 0:
            logger.warning(f"{skipped} query dilewati karena rate limit atau error.")
        
        return responses
    
    def evaluate_llm_all_models(self) -> Dict:
        """
        Evaluasi semua model Groq yang tersedia.
        
        Returns:
            Dict dengan hasil evaluasi per model
        """
        logger.info("=" * 60)
        logger.info("Evaluating ALL Groq models...")
        
        compliance_tests = self.test_dataset.get("compliance_tests", [])
        
        if not compliance_tests:
            logger.warning("No compliance tests found")
            return {}
        
        results = {}
        
        # Get list of models from config
        models_to_eval = list(GROQ_ALLOWED_MODELS)
        logger.info(f"Models to evaluate: {len(models_to_eval)}")
        
        # Store original model
        original_model = CONFIG.get("groq_model", "llama-3.3-70b-versatile")
        
        for model_id in models_to_eval:
            logger.info(f"\n--- Evaluating model: {model_id} ---")
            
            try:
                # Switch model
                from config import set_groq_model
                set_groq_model(model_id)
                
                # Update pipeline's groq client? Pipeline uses config directly
                # The pipeline's models.groq_client is already instantiated with original API key
                # We need to recreate the groq_client with new model? No, just config change works.
                
                # Evaluate throughput (using official data)
                throughput = self.llm_eval.evaluate_throughput()
                
                # Evaluate compliance
                compliance = None
                if compliance_tests:
                    try:
                        compliance = _call_with_rate_limit_retry(
                            self.llm_eval.evaluate_instruction_compliance, 
                            compliance_tests, 
                            max_retries=2
                        )
                    except Exception as e:
                        logger.error(f"Compliance evaluation for {model_id} failed: {e}")
                        compliance = None
                
                # Evaluate conciseness (using gold answers)
                conciseness = []
                if self.test_dataset and self.test_dataset.get("rag_queries"):
                    test_responses = [
                        {"question": q.get("question", ""), "answer": q.get("gold_answer", "")}
                        for q in self.test_dataset.get("rag_queries", [])[:10]
                        if q.get("gold_answer")
                    ]
                    if test_responses:
                        conciseness = self.llm_eval.evaluate_conciseness(test_responses)
                
                results[model_id] = {
                    "throughput": throughput,
                    "compliance": compliance,
                    "conciseness": conciseness
                }
                
                logger.info(f"✓ {model_id} evaluation complete")
                
                # Wait a bit between models to avoid rate limits
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_id}: {e}")
                results[model_id] = {"error": str(e)}
        
        # Restore original model
        try:
            set_groq_model(original_model)
        except Exception:
            pass
        
        return results
    
    def run_full_evaluation(
        self,
        test_dataset_path: str,
        output_dir: str = "output",
        skip_embedder: bool = False,
        skip_rag: bool = False,
        skip_llm: bool = False,
    ) -> Dict:
        self.load_test_dataset(test_dataset_path)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "embedder": {},
            "rag": {},
            "llm": {},
        }
        
        config = {
            "groq_model": CONFIG["groq_model"],
            "embedding_model": CONFIG["embedding_model"],
            "reranker_model": CONFIG["reranker_model"],
            "chroma_retrieval_k": CONFIG["chroma_retrieval_k"],
            "reranked_k": CONFIG["reranked_k"],
            "context_window": CONFIG["context_window"]
        }
        
        # ── Embedder Evaluation (Graph vs Raw) ────────────────────────────────
        if not skip_embedder:
            logger.info("=" * 60)
            logger.info("Starting Embedder Evaluation...")
            results["embedder"] = self.evaluate_embedder()
            
            if results["embedder"]:
                # Save PNG plot
                embedder_plot_path = os.path.join(
                    output_dir, "embedder", 
                    f"{self.timestamp}_embedder_metrics.png"
                )
                os.makedirs(os.path.dirname(embedder_plot_path), exist_ok=True)
                
                self.embedder_eval.plot_metrics(
                    graph_vs_raw_metrics=results["embedder"].get("graph_vs_raw", {}),
                    save_path=embedder_plot_path
                )
                
                # Save JSON metrics
                json_path = os.path.join(
                    output_dir, "embedder",
                    f"{self.timestamp}_embedder_metrics.json"
                )
                self._save_json(results["embedder"], json_path)
                
                logger.info(f"Embedder plot saved to: {embedder_plot_path}")
        
        # ── RAG Evaluation (Graph vs Raw comparison) ─────────────────────────
        if not skip_rag:
            logger.info("=" * 60)
            logger.info("Starting RAG Evaluation (Graph vs Raw)...")
            results["rag"] = self.evaluate_rag_both_pipelines()
            
            if results["rag"]:
                # Save PNG plot
                rag_plot_path = os.path.join(
                    output_dir, "rag",
                    f"{self.timestamp}_rag_comparison.png"
                )
                os.makedirs(os.path.dirname(rag_plot_path), exist_ok=True)
                
                self.rag_eval.plot_comparison_metrics(
                    graph_metrics=results["rag"].get("graph_rag", {}),
                    raw_metrics=results["rag"].get("raw_rag", {}),
                    save_path=rag_plot_path
                )
                
                # Save JSON metrics
                json_path = os.path.join(
                    output_dir, "rag",
                    f"{self.timestamp}_rag_metrics.json"
                )
                self._save_json(results["rag"], json_path)
                
                logger.info(f"RAG comparison plot saved to: {rag_plot_path}")
        
        # ── LLM Evaluation (all models) ──────────────────────────────────────
        if not skip_llm:
            logger.info("=" * 60)
            logger.info("Starting LLM Evaluation (ALL models)...")
            results["llm"] = self.evaluate_llm_all_models()
            
            if results["llm"]:
                # Save PNG plot
                llm_plot_path = os.path.join(
                    output_dir, "model",
                    f"{self.timestamp}_llm_comparison.png"
                )
                os.makedirs(os.path.dirname(llm_plot_path), exist_ok=True)
                
                self.llm_eval.plot_all_models_metrics(
                    all_models_metrics=results["llm"],
                    save_path=llm_plot_path
                )
                
                # Save JSON metrics
                json_path = os.path.join(
                    output_dir, "model",
                    f"{self.timestamp}_llm_metrics.json"
                )
                self._save_json(results["llm"], json_path)
                
                logger.info(f"LLM comparison plot saved to: {llm_plot_path}")
        
        return results
    
    def _save_json(self, data: Dict, path: str):
        """Simpan data ke file JSON."""
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
            if hasattr(obj, 'score'):
                return float(obj.score)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert)
    
    def print_summary(self, results: Dict):
        """Print ringkasan hasil evaluasi ke console."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        # Embedder summary (Graph vs Raw only)
        embedder = results.get("embedder", {})
        graph_raw = embedder.get("graph_vs_raw", {})
        improvement = graph_raw.get("improvement", {})
        
        print(f"\n🔍 EMBEDDER (Graph vs Raw):")
        print(f"   Context Length: +{improvement.get('context_length_pct', 0):.1f}%")
        print(f"   Entity Coverage: +{improvement.get('entity_coverage_pct', 0):.1f}%")
        print(f"   Semantic Similarity: +{improvement.get('semantic_similarity_pct', 0):.1f}%")
        
        # RAG comparison summary
        rag = results.get("rag", {})
        
        print(f"\n🤖 RAG COMPARISON:")
        
        # Graph RAG
        graph_rag = rag.get("graph_rag", {})
        if graph_rag:
            f_scores = [f.score for f in graph_rag.get("faithfulness", [])]
            c_scores = [c.score for c in graph_rag.get("completeness", [])]
            r_scores = [r.score for r in graph_rag.get("relevance", [])]
            speed = graph_rag.get("speed", {})
            
            print(f"   Graph RAG:")
            print(f"     Faithfulness: {np.mean(f_scores):.3f}" if f_scores else "     Faithfulness: N/A")
            print(f"     Completeness: {np.mean(c_scores):.3f}" if c_scores else "     Completeness: N/A")
            print(f"     Answer Relev: {np.mean(r_scores):.3f}" if r_scores else "     Answer Relev: N/A")
            print(f"     Response Time: {speed.get('total', {}).get('mean', 0):.2f}s")
        
        # Raw RAG
        raw_rag = rag.get("raw_rag", {})
        if raw_rag:
            f_scores = [f.score for f in raw_rag.get("faithfulness", [])]
            c_scores = [c.score for c in raw_rag.get("completeness", [])]
            r_scores = [r.score for r in raw_rag.get("relevance", [])]
            speed = raw_rag.get("speed", {})
            
            print(f"   Raw RAG:")
            print(f"     Faithfulness: {np.mean(f_scores):.3f}" if f_scores else "     Faithfulness: N/A")
            print(f"     Completeness: {np.mean(c_scores):.3f}" if c_scores else "     Completeness: N/A")
            print(f"     Answer Relev: {np.mean(r_scores):.3f}" if r_scores else "     Answer Relev: N/A")
            print(f"     Response Time: {speed.get('total', {}).get('mean', 0):.2f}s")
        
        # LLM all models summary
        llm = results.get("llm", {})
        
        print(f"\n🧠 LLM (All Models):")
        for model_name, model_results in llm.items():
            if "error" in model_results:
                print(f"   {model_name}: ❌ {model_results['error']}")
                continue
            
            throughput = model_results.get("throughput", {})
            compliance = model_results.get("compliance")
            conciseness = model_results.get("conciseness", [])
            
            tps = throughput.get("overall_prompts", {}).get("mean_tps", 0) if throughput else 0
            comp_score = compliance.overall_score if compliance else 0
            conc_score = np.mean([c.score for c in conciseness]) if conciseness else 0
            
            status = "✅" if comp_score > 0.7 else "⚠️"
            print(f"   {model_name[:30]}: TPS={tps:.0f} | Comp={comp_score:.2f} | Conc={conc_score:.2f} {status}")
        
        print("=" * 70)
        print(f"\n📄 Output saved to: output/embedder/, output/rag/, output/model/")