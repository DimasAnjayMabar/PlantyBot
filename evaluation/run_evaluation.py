# scripts/run_evaluation.py
"""
Script terpisah untuk menjalankan evaluasi.
Dijalankan hanya saat diperlukan, tidak otomatis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import get_rag_pipeline
from evaluation.main import RAGEvaluationPipeline

def main():
    print("=" * 60)
    print("RAG PIPELINE EVALUATOR (Independent Mode)")
    print("=" * 60)
    
    # 1. Inisialisasi pipeline (normal)
    print("\n[1] Initializing RAG Pipeline...")
    pipeline = get_rag_pipeline()
    
    # 2. Inisialisasi evaluator (pinjam models dari pipeline)
    print("[2] Initializing Evaluator...")
    evaluator = RAGEvaluationPipeline(pipeline)
    
    # 3. Jalankan evaluasi (hanya saat diminta)
    print("[3] Running Evaluation...")
    results = evaluator.run_full_evaluation(
        test_dataset_path="test_dataset.json",
        output_dir="output",
        skip_embedder=False,
        skip_rag=False,
        skip_llm=False
    )
    
    # 4. Tampilkan ringkasan
    evaluator.print_summary(results)
    
    print("\n✅ Evaluation complete! Reports saved to output")

if __name__ == "__main__":
    main()