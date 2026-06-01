# evaluation/__init__.py
"""
Sistem Evaluasi untuk RAG Pipeline Agribot

Sub-modules:
- embedder_eval: Evaluasi performa embedding & retrieval
- rag_eval: Evaluasi faithfulness, completeness, answer relevance
- llm_eval: Evaluasi throughput, instruction compliance, conciseness
- report_generator: Generate PDF dengan tabel dan visualisasi
"""

from .embedder_eval import EmbedderEvaluator
from .rag_eval import RAGEvaluator
from .llm_eval import LLMEvaluator
from .report_generator import ReportGenerator
from .main import RAGEvaluationPipeline

__all__ = [
    'EmbedderEvaluator',
    'RAGEvaluator', 
    'LLMEvaluator',
    'ReportGenerator',
    'RAGEvaluationPipeline'
]