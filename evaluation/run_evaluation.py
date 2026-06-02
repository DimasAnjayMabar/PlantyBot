# scripts/run_evaluation.py - VERSI REVISI

"""
Script terpisah untuk menjalankan evaluasi.
Output: PNG plots langsung di folder masing-masing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import get_rag_pipeline
from evaluation.main import RAGEvaluationPipeline


def main():
    print("=" * 60)
    print("RAG PIPELINE EVALUATOR (Full Suite)")
    print("=" * 60)

    print("\n[1] Initializing RAG Pipeline...")
    pipeline = get_rag_pipeline()

    print("[2] Initializing Evaluator...")
    evaluator = RAGEvaluationPipeline(pipeline)

    print("[3] Running Full Evaluation...")
    print("    - Embedder: Graph vs Raw comparison")
    print("    - RAG: Graph RAG vs Raw RAG comparison")
    print("    - LLM: All Groq models evaluation")
    print()
    
    results = evaluator.run_full_evaluation(
        test_dataset_path="test_dataset.json",
        output_dir="output",
        skip_embedder=True,
        skip_rag=False,
        skip_llm=True
    )

    print("\n[4] Summary:")
    evaluator.print_summary(results)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    print("\n📁 Output files:")
    print("   output/embedder/     - Embedder PNG plot + JSON metrics")
    print("   output/rag/          - RAG comparison PNG plot + JSON metrics")
    print("   output/model/        - LLM all models PNG plot + JSON metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()