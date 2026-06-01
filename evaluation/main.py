# evaluation/main.py
"""
Main evaluator yang mengintegrasikan semua komponen.
"""

import os
import json
import logging
import sys
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

from .embedder_eval import EmbedderEvaluator
from .rag_eval import RAGEvaluator
from .llm_eval import LLMEvaluator
from .report_generator import ReportGenerator

from config import CONFIG

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class RAGEvaluationPipeline:
    """
    Main orchestrator untuk menjalankan evaluasi lengkap.
    
    Usage:
        evaluator = RAGEvaluationPipeline(rag_pipeline)
        
        # Run all evaluations
        results = evaluator.run_full_evaluation(
            test_dataset_path="./evaluation/test_dataset.json"
        )
        
        # Or run individual components
        embedder_results = evaluator.evaluate_embedder(test_queries)
        rag_results = evaluator.evaluate_rag(test_queries)
        llm_results = evaluator.evaluate_llm(test_prompts)
    """
    
    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: Instance RAGPipeline yang sudah running
        """
        self.pipeline = rag_pipeline
        self.embedder_eval = EmbedderEvaluator(rag_pipeline)
        self.rag_eval = RAGEvaluator(rag_pipeline)
        self.llm_eval = LLMEvaluator(rag_pipeline)
        self.report_gen = ReportGenerator()
        
        self.test_dataset = None
    
    def load_test_dataset(self, dataset_path: str) -> Dict:
        """
        Load test dataset dari file JSON.
        
        Expected format:
        {
            "embedder_queries": [...],
            "rag_queries": [...],
            "llm_prompts": [...],
            "compliance_tests": [...],
            "causal_ground_truth": {...}
        }
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.test_dataset = json.load(f)
        logger.info(f"Loaded test dataset from {dataset_path}")
        return self.test_dataset
    
    def evaluate_embedder(self, queries: List[Dict] = None) -> Dict:
        """
        Evaluasi komponen embedder.
        """
        if queries is None:
            queries = self.test_dataset.get("embedder_queries", [])
        
        if not queries:
            logger.warning("No embedder test queries found")
            return {}
        
        logger.info(f"Running embedder evaluation on {len(queries)} queries...")
        
        retrieval_metrics = self.embedder_eval.evaluate_retrieval_accuracy(queries)
        graph_vs_raw = self.embedder_eval.compare_graph_vs_raw(queries)
        diversity = self.embedder_eval.evaluate_context_diversity(queries)
        
        return {
            "retrieval_metrics": retrieval_metrics,
            "graph_vs_raw": graph_vs_raw,
            "diversity_metrics": diversity
        }
    
    def evaluate_rag(self, queries: List[Dict] = None) -> Dict:
        """
        Evaluasi komponen RAG (end-to-end dengan generation).
        """
        if queries is None:
            queries = self.test_dataset.get("rag_queries", [])
        
        if not queries:
            logger.warning("No RAG test queries found")
            return {}
        
        logger.info(f"Running RAG evaluation on {len(queries)} queries...")
        
        # Execute queries and collect responses
        responses = []
        for q in queries:
            question = q.get("question", "")
            if not question:
                continue
            
            # Process query through pipeline
            response = self.pipeline.process_query(
                question,
                chat_id=q.get("chat_id"),
                user_id=q.get("user_id")
            )
            
            # Collect answer (non-streaming)
            answer_text = ""
            if hasattr(response.answer, '__iter__') and not isinstance(response.answer, str):
                for chunk in response.answer:
                    answer_text += chunk
            else:
                answer_text = str(response.answer)
            
            responses.append({
                "question": question,
                "answer": answer_text,
                "chunks": response.final_chunks,
                "processing_time": response.processing_time,
                "gold_answer": q.get("gold_answer"),
                "expected_causal": q.get("expected_causal_relations", [])
            })
        
        # Evaluate metrics
        faithfulness = self.rag_eval.evaluate_faithfulness_batch(responses)
        completeness = self.rag_eval.evaluate_completeness_batch(
            responses, 
            self.test_dataset.get("causal_ground_truth", {})
        )
        relevance = self.rag_eval.evaluate_answer_relevance_batch(responses)
        speed = self.rag_eval.evaluate_speed([q["question"] for q in queries if q.get("question")])
        
        return {
            "faithfulness": faithfulness,
            "completeness": completeness,
            "relevance": relevance,
            "speed": speed,
            "responses": responses  # For debugging
        }
    
    def evaluate_llm(self, prompts: List[str] = None, compliance_tests: List[Dict] = None) -> Dict:
        """
        Evaluasi komponen LLM secara terpisah.
        """
        if prompts is None:
            prompts = self.test_dataset.get("llm_prompts", [])
        
        if compliance_tests is None:
            compliance_tests = self.test_dataset.get("compliance_tests", [])
        
        logger.info(f"Running LLM evaluation on {len(prompts)} prompts and {len(compliance_tests)} compliance tests...")
        
        throughput = self.llm_eval.evaluate_throughput(prompts) if prompts else {}
        compliance = self.llm_eval.evaluate_instruction_compliance(compliance_tests) if compliance_tests else None
        
        # Test conciseness with RAG responses (if available)
        conciseness = []
        if self.test_dataset and self.test_dataset.get("rag_queries"):
            # Use some RAG responses or create simple test
            test_responses = [
                {"question": q.get("question", ""), "answer": q.get("gold_answer", "")}
                for q in self.test_dataset.get("rag_queries", [])[:10]
                if q.get("gold_answer")
            ]
            if test_responses:
                conciseness = self.llm_eval.evaluate_conciseness(test_responses)
        
        return {
            "throughput": throughput,
            "compliance": compliance,
            "conciseness": conciseness
        }
    
    def run_full_evaluation(
        self,
        test_dataset_path: str,
        output_dir: str = "output",
        skip_embedder: bool = False,
        skip_rag: bool = False,
        skip_llm: bool = False,
        generate_combined: bool = True
    ) -> Dict:
        """
        Jalankan semua evaluasi dan generate report.
        
        Args:
            test_dataset_path: Path ke file JSON test dataset
            output_dir: Base output directory
            skip_embedder: Skip embedder evaluation
            skip_rag: Skip RAG evaluation
            skip_llm: Skip LLM evaluation
            generate_combined: Generate combined report
        
        Returns:
            Dictionary dengan semua metrics dan report paths
        """
        # Load test dataset
        self.load_test_dataset(test_dataset_path)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "embedder": {},
            "rag": {},
            "llm": {},
            "reports": {}
        }
        
        config = {
            "groq_model": CONFIG["groq_model"],
            "embedding_model": CONFIG["embedding_model"],
            "reranker_model": CONFIG["reranker_model"],
            "chroma_retrieval_k": CONFIG["chroma_retrieval_k"],
            "reranked_k": CONFIG["reranked_k"],
            "context_window": CONFIG["context_window"]
        }
        
        # ── Embedder Evaluation ──────────────────────────────────────────────
        if not skip_embedder:
            logger.info("=" * 60)
            logger.info("Starting Embedder Evaluation...")
            results["embedder"] = self.evaluate_embedder()
            
            # Generate embedder report
            report_path = self.report_gen.generate_embedder_report(
                retrieval_metrics=results["embedder"].get("retrieval_metrics", {}),
                graph_vs_raw_metrics=results["embedder"].get("graph_vs_raw", {}),
                diversity_metrics=results["embedder"].get("diversity_metrics", {}),
                config=config
            )
            results["reports"]["embedder"] = report_path
            logger.info(f"Embedder report saved to: {report_path}")
        
        # ── RAG Evaluation ───────────────────────────────────────────────────
        if not skip_rag:
            logger.info("=" * 60)
            logger.info("Starting RAG Evaluation...")
            results["rag"] = self.evaluate_rag()
            
            # Generate RAG report
            report_path = self.report_gen.generate_rag_report(
                faithfulness_metrics=results["rag"].get("faithfulness", []),
                completeness_metrics=results["rag"].get("completeness", []),
                relevance_metrics=results["rag"].get("relevance", []),
                speed_metrics=results["rag"].get("speed", {}),
                config=config
            )
            results["reports"]["rag"] = report_path
            logger.info(f"RAG report saved to: {report_path}")
        
        # ── LLM Evaluation ───────────────────────────────────────────────────
        if not skip_llm:
            logger.info("=" * 60)
            logger.info("Starting LLM Evaluation...")
            results["llm"] = self.evaluate_llm()
            
            # Generate model report
            report_path = self.report_gen.generate_model_report(
                throughput_metrics=results["llm"].get("throughput", {}),
                compliance_metrics=results["llm"].get("compliance"),
                conciseness_metrics=results["llm"].get("conciseness", []),
                config=config
            )
            results["reports"]["model"] = report_path
            logger.info(f"Model report saved to: {report_path}")
        
        # ── Combined Report ───────────────────────────────────────────────────
        if generate_combined and results["embedder"] and results["rag"] and results["llm"]:
            logger.info("=" * 60)
            logger.info("Generating Combined Report...")
            
            combined_path = self.report_gen.generate_combined_report(
                all_metrics=results,
                config=config
            )
            results["reports"]["combined"] = combined_path
            logger.info(f"Combined report saved to: {combined_path}")
        
        # Save master results JSON
        master_json_path = os.path.join(output_dir, "combined", f"{self.report_gen.timestamp}_master_results.json")
        self.report_gen._save_json(results, master_json_path)
        
        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info(f"Reports saved to: {output_dir}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print ringkasan hasil evaluasi ke console."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        # Embedder summary
        embedder = results.get("embedder", {})
        retrieval = embedder.get("retrieval_metrics", {})
        recall_5 = retrieval.get("overall", {}).get("recall@5", {}).get("mean", 0)
        mrr = retrieval.get("avg_mrr", 0)
        
        print(f"\n🔍 EMBEDDER:")
        print(f"   Recall@5:     {recall_5:.3f}")
        print(f"   MRR:          {mrr:.3f}")
        
        graph_raw = embedder.get("graph_vs_raw", {})
        improvement = graph_raw.get("improvement", {})
        print(f"   Graph vs Raw: +{improvement.get('context_length_pct', 0):.1f}% length, "
              f"+{improvement.get('entity_coverage_pct', 0):.1f}% entities")
        
        # RAG summary
        rag = results.get("rag", {})
        faithfulness = rag.get("faithfulness", [])
        completeness = rag.get("completeness", [])
        relevance = rag.get("relevance", [])
        speed = rag.get("speed", {})
        
        f_score = np.mean([f.score for f in faithfulness]) if faithfulness else 0
        c_score = np.mean([c.score for c in completeness]) if completeness else 0
        r_score = np.mean([r.score for r in relevance]) if relevance else 0
        
        print(f"\n🤖 RAG:")
        print(f"   Faithfulness: {f_score:.3f}")
        print(f"   Completeness: {c_score:.3f}")
        print(f"   Answer Relev: {r_score:.3f}")
        print(f"   Response Time: {speed.get('total', {}).get('mean', 0):.2f}s")
        
        # LLM summary
        llm = results.get("llm", {})
        throughput = llm.get("throughput", {})
        compliance = llm.get("compliance")
        conciseness = llm.get("conciseness", [])
        
        tps = throughput.get("overall_prompts", {}).get("mean_tps", 0) if throughput else 0
        comp_score = compliance.overall_score if compliance else 0
        conc_score = np.mean([c.score for c in conciseness]) if conciseness else 0
        
        print(f"\n🧠 LLM:")
        print(f"   Tokens/sec:   {tps:.1f}")
        print(f"   Compliance:   {comp_score:.3f}")
        print(f"   Conciseness:  {conc_score:.3f}")
        
        # Overall
        overall = np.mean([recall_5, f_score, c_score, r_score, comp_score, conc_score])
        print(f"\n⭐ OVERALL SCORE: {overall:.3f}")
        print("=" * 70)
        
        # Report paths
        print("\n📄 Reports generated:")
        for name, path in results.get("reports", {}).items():
            print(f"   {name}: {path}")