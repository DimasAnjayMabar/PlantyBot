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

logger = logging.getLogger("ragna")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))  # folder /evaluation
SAVE_DATA_DIR = os.path.join(EVAL_DIR, "save_data")

import matplotlib
matplotlib.use('Agg')

# ── Rate-limit helper ─────────────────────────────────────────────────────────
class TPDLimitExceeded(Exception):
    """Rate limit dengan scope harian (tokens per day) — retry dalam
    hitungan menit tidak akan membantu, kuota baru pulih besok."""
    pass


def _is_tpd_error(error_message: str) -> bool:
    msg = error_message.lower()
    return "per day" in msg or "(tpd)" in msg

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
            error_str = str(e)
            if _is_tpd_error(error_str):
                logger.error(
                    "Groq TPD (tokens per day) limit tercapai. Tidak retry — "
                    "kuota baru pulih besok. Progres sudah tersimpan di checkpoint."
                )
                raise TPDLimitExceeded(error_str) from e

            wait_sec = _parse_retry_after(error_str)
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit (attempt {attempt+1}/{max_retries}). Menunggu {wait_sec:.0f}s...")
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
        question_list = [q["question"] for q in queries if q.get("question")]

        # ── Graph RAG (improved) ──────────────────────────────────────────────
        logger.info("  Evaluating GRAPH RAG pipeline...")
        graph_responses = self._evaluate_pipeline_responses(
            queries, use_raw=False
        )

        # ── Raw RAG (regular) ────────────────────────────────────────────────
        logger.info("  Evaluating RAW RAG pipeline...")
        original_mode = CONFIG.get("rag_mode", "improved")
        CONFIG["rag_mode"] = "regular"

        raw_responses = self._evaluate_pipeline_responses(
            queries, use_raw=True
        )

        CONFIG["rag_mode"] = original_mode

        if not graph_responses and not raw_responses:
            logger.error("Both pipelines failed.")
            return {}

        results = {"graph_rag": {}, "raw_rag": {}}

        def _build_speed_dict(responses: List[Dict]) -> Dict:
            total_times      = [r["processing_time"] for r in responses if r.get("processing_time")]
            gen_times        = [r["generation_time"] for r in responses if r.get("generation_time")]
            retrieval_times  = [
                r.get("retrieval_time", 0.0) + r.get("enrichment_time", 0.0)
                for r in responses
            ]
            rerank_times     = [r.get("rerank_time", 0.0) for r in responses]

            def _safe_stat(values):
                if not values:
                    return {"mean": 0.0, "std": 0.0}
                return {"mean": float(np.mean(values)), "std": float(np.std(values))}

            return {
                "retrieval": _safe_stat(retrieval_times),
                "rerank": _safe_stat(rerank_times),
                "generation": _safe_stat(gen_times),
                "total": _safe_stat(total_times),
            }

        if graph_responses:
            faithfulness = self.rag_eval.evaluate_faithfulness_batch(graph_responses)
            completeness = self.rag_eval.evaluate_completeness_batch(
                graph_responses, self.test_dataset.get("causal_ground_truth", {})
            )
            relevance = self.rag_eval.evaluate_answer_relevance_batch(graph_responses)
            speed = _build_speed_dict(graph_responses)   # <-- FIX: 1 argumen saja

            results["graph_rag"] = {
                "faithfulness": faithfulness, "completeness": completeness,
                "relevance": relevance, "speed": speed, "responses": graph_responses
            }

        if raw_responses:
            faithfulness = self.rag_eval.evaluate_faithfulness_batch(raw_responses)
            completeness = self.rag_eval.evaluate_completeness_batch(
                raw_responses, self.test_dataset.get("causal_ground_truth", {})
            )
            relevance = self.rag_eval.evaluate_answer_relevance_batch(raw_responses)
            speed = _build_speed_dict(raw_responses)     # <-- FIX: 1 argumen saja

            results["raw_rag"] = {
                "faithfulness": faithfulness, "completeness": completeness,
                "relevance": relevance, "speed": speed, "responses": raw_responses
            }

        return results
    
    def _evaluate_pipeline_responses(
        self, 
        queries: List[Dict], 
        use_raw: bool = False
    ) -> List[Dict]:
        pipeline_label = "RAW" if use_raw else "GRAPH"
        checkpoint_dir = os.path.join(SAVE_DATA_DIR, "rag")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"_checkpoint_{'raw' if use_raw else 'graph'}_responses.json"
        )

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            query_by_question = {q.get("question", ""): q for q in queries}
            responses = []
            for r in loaded:
                q_text = r["question"]
                if q_text in query_by_question:
                    current_q = query_by_question[q_text]
                    r["gold_answer"] = current_q.get("gold_answer")
                    r["expected_causal"] = current_q.get("expected_causal_relations", [])
                    responses.append(r)

            if len(responses) != len(loaded):
                logger.warning(
                    f"  {len(loaded) - len(responses)} entri checkpoint dibuang "
                    f"(pertanyaan tidak lagi ada di test_dataset.json saat ini)."
                )
            logger.info(f"  Resuming: {len(responses)} response valid dari checkpoint.")
        else:
            responses = []

        done_questions = {r["question"] for r in responses}
        skipped = 0

        for idx, q in enumerate(queries):
            question = q.get("question", "")
            if not question or question in done_questions:
                continue

            # ── PENANDA MULAI QUERY ─────────────────────────────────────
            logger.info(
                f"=== [{pipeline_label}] Query {idx + 1}/{len(queries)} START: "
                f"{question[:60]}..."
            )
            q_start = time.perf_counter()

            def _run_query():
                if use_raw:
                    original = CONFIG.get("rag_mode", "improved")
                    CONFIG["rag_mode"] = "regular"
                    try:
                        return self.pipeline.process_query(
                            question, chat_id=q.get("chat_id"), user_id=q.get("user_id")
                        )
                    finally:
                        CONFIG["rag_mode"] = original
                else:
                    return self.pipeline.process_query(
                        question, chat_id=q.get("chat_id"), user_id=q.get("user_id")
                    )

            try:
                response = _call_with_rate_limit_retry(_run_query, max_retries=3)
            except TPDLimitExceeded:
                logger.error(
                    f"=== [{pipeline_label}] STOP: TPD limit tercapai di Query {idx + 1}/{len(queries)}. "
                    f"{len(responses)} response sudah aman di checkpoint. "
                    f"Jalankan ulang script besok untuk melanjutkan otomatis."
                )
                return responses   # <- keluar total dari fungsi, bukan lanjut loop
            except Exception as e:
                logger.error(f"=== [{pipeline_label}] Query {idx + 1}/{len(queries)} FAILED: {e}")
                skipped += 1
                continue
            except Exception as e:
                logger.error(f"=== [{pipeline_label}] Query {idx + 1}/{len(queries)} FAILED: {e}")
                skipped += 1
                continue
                
            # ── PENANDA: mulai membaca stream generation ────────────────
            logger.info(
                f"    [{pipeline_label}] Query {idx + 1}: retrieval+enrich+rerank "
                f"selesai, mulai membaca generation stream..."
            )
            gen_start = time.perf_counter()

            answer_text = ""
            try:
                if hasattr(response.answer, '__iter__') and not isinstance(response.answer, str):
                    for chunk in response.answer:
                        answer_text += chunk
                else:
                    answer_text = str(response.answer)
            except Exception as e:
                from groq import RateLimitError
                if isinstance(e, RateLimitError) and _is_tpd_error(str(e)):
                    logger.error(
                        f"=== [{pipeline_label}] TPD limit tercapai di Query {idx + 1}/{len(queries)}. "
                        f"Menghentikan evaluasi sekarang."
                    )
                    raise TPDLimitExceeded(str(e)) from e
                elif isinstance(e, RateLimitError):
                    wait_sec = _parse_retry_after(str(e))
                    logger.warning(f"    [{pipeline_label}] Query {idx+1}: rate limit. Menunggu {wait_sec:.0f}s...")
                    time.sleep(wait_sec + 5)
                    try:
                        response = _run_query()
                        if hasattr(response.answer, '__iter__') and not isinstance(response.answer, str):
                            for chunk in response.answer:
                                answer_text += chunk
                        else:
                            answer_text = str(response.answer)
                    except Exception as e2:
                        logger.error(f"=== [{pipeline_label}] Query {idx + 1} retry stream gagal: {e2}")
                        skipped += 1
                        continue
                else:
                    logger.error(f"=== [{pipeline_label}] Query {idx+1} error: {e}")
                    skipped += 1
                    continue

            gen_elapsed = time.perf_counter() - gen_start
            total_elapsed = time.perf_counter() - q_start

            # ── PENANDA SELESAI QUERY, dengan breakdown waktu ───────────
            logger.info(
                f"=== [{pipeline_label}] Query {idx + 1}/{len(queries)} DONE | "
                f"generation_stream={gen_elapsed:.1f}s | "
                f"total_cycle={total_elapsed:.1f}s | "
                f"pipeline.processing_time={response.processing_time:.1f}s"
            )

            responses.append({
                "question": question,
                "answer": answer_text,
                "chunks": response.final_chunks,
                "processing_time": response.processing_time,
                "generation_time": gen_elapsed,
                "retrieval_time": response.retrieval_time,      # <-- TAMBAHKAN
                "enrichment_time": response.enrichment_time,     # <-- TAMBAHKAN
                "rerank_time": response.rerank_time,             # <-- TAMBAHKAN
                "gold_answer": q.get("gold_answer"),
                "expected_causal": q.get("expected_causal_relations", [])
            })

            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=2, ensure_ascii=False, default=str)

        if skipped > 0:
            logger.warning(f"[{pipeline_label}] {skipped} query dilewati karena rate limit atau error.")

        return responses
    
    def evaluate_llm_all_models(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Evaluating ALL Groq models...")

        compliance_tests = self.test_dataset.get("compliance_tests", [])
        if not compliance_tests:
            logger.warning("No compliance tests found")
            return {}

        llm_prompts = self.test_dataset.get("llm_prompts", [])
        if not llm_prompts:
            logger.warning("No llm_prompts found — conciseness evaluation will be skipped")

        checkpoint_dir = os.path.join(SAVE_DATA_DIR, "llm")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "_checkpoint_llm_results.json")

        results = {}
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"  Resuming: {len(results)} model sudah selesai dari checkpoint.")

        models_to_eval = list(GROQ_ALLOWED_MODELS)
        logger.info(f"Models to evaluate: {len(models_to_eval)}")

        original_model = CONFIG.get("groq_model", "llama-3.3-70b-versatile")

        for model_id in models_to_eval:
            if model_id in results and "error" not in results[model_id]:
                logger.info(f"  Skip {model_id} (sudah ada di checkpoint)")
                continue

            logger.info(f"\n--- Evaluating model: {model_id} ---")
            try:
                from config import set_groq_model
                from dataclasses import asdict
                set_groq_model(model_id)

                throughput = self.llm_eval.evaluate_throughput()

                compliance = None
                if compliance_tests:
                    try:
                        compliance = _call_with_rate_limit_retry(
                            self.llm_eval.evaluate_instruction_compliance,
                            compliance_tests, max_retries=2
                        )
                    except Exception as e:
                        logger.error(f"Compliance evaluation for {model_id} failed: {e}")
                        compliance = None

                # ── Conciseness — generate jawaban ASLI dari llm_prompts, ──────
                # ── bukan mengevaluasi gold_answer yang ditulis manual ─────────
                conciseness = []
                if llm_prompts:
                    test_responses = []
                    for prompt in llm_prompts:
                        try:
                            def _generate_for_prompt():
                                resp = self.rag_eval.models.groq_client.chat.completions.create(
                                    model=CONFIG["groq_model"],
                                    messages=[{"role": "user", "content": prompt}],
                                    max_tokens=512,
                                    temperature=0.2,
                                )
                                return resp.choices[0].message.content

                            answer = _call_with_rate_limit_retry(_generate_for_prompt, max_retries=2)
                            test_responses.append({"question": prompt, "answer": answer})
                        except Exception as e:
                            logger.error(f"  Conciseness generation gagal untuk prompt '{prompt[:40]}...': {e}")
                            continue

                    if test_responses:
                        conciseness = self.llm_eval.evaluate_conciseness(test_responses)

                # ── FIX: konversi dataclass -> dict SEBELUM disimpan ke JSON ───
                # (dataclass mentah gagal di-serialize dengan benar, jatuh ke
                # str() lewat default=str dan tidak bisa dibaca ulang saat resume)
                compliance_dict = asdict(compliance) if compliance else None
                conciseness_list = [asdict(c) for c in conciseness] if conciseness else []

                results[model_id] = {
                    "throughput": throughput,
                    "compliance": compliance_dict,
                    "conciseness": conciseness_list
                }
                logger.info(f"✓ {model_id} evaluation complete")

                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

                time.sleep(10)

            except Exception as e:
                logger.error(f"Failed to evaluate {model_id}: {e}")
                results[model_id] = {"error": str(e)}
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

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
            
            # FIX: compliance sekarang adalah dict, bukan dataclass
            comp_score = compliance.get("overall_score", 0) if compliance else 0
            
            # FIX: conciseness adalah list of dict, bukan dataclass
            conc_score = np.mean([c.get("score", 0) for c in conciseness]) if conciseness else 0
            
            status = "✅" if comp_score > 0.7 else "⚠️"
            print(f"   {model_name[:30]}: TPS={tps:.0f} | Comp={comp_score:.2f} | Conc={conc_score:.2f} {status}")
        
        print("=" * 70)
        print(f"\n📄 Output saved to: output/embedder/, output/rag/, output/model/")