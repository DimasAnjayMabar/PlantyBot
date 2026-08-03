import os
import queue as _queue
import logging
import threading
import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

import base64
import io
from PIL import Image
import google.generativeai as genai

import torch
from groq import Groq
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer as AutoTok,
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from config import CONFIG, PROMPTS, set_llm_mode, list_local_models, GROQ_MODEL_SAFE_TOKEN_BUDGET, FIXED_OVERHEAD_TOKENS
load_dotenv()

############################################################
# LOGGING SETUP
############################################################

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ragna")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    GREY   = "\x1b[38;5;245m"
    CYAN   = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RED    = "\x1b[31m"
    BOLD   = "\x1b[1m"
    RESET  = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD + RED,
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = LEVEL_COLORS.get(record.levelno, RESET)
            record.levelname = f"{color}{record.levelname:<8}{RESET}"
            record.name      = f"{GREY}{record.name}{RESET}"
            return super().format(record)

    console_fmt = ColorFormatter(
        fmt="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(console_fmt)

    fh = logging.FileHandler("ragna.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = _setup_logger()

############################################################
# DATA STRUCTURES
############################################################

@dataclass
class CandidateChunk:
    """Hasil dari ChromaDB wide retrieval (Tahap 1)."""
    isi_id:       str    # id Node Isi di Neo4j (kunci relasi)
    jurnal_id:    str    # id Node Jurnal
    konten_chunk: str    # teks chunk mentah
    vector_score: float  # jarak kosinus ChromaDB (lebih kecil = lebih dekat)


@dataclass
class EnrichedChunk:
    """Hasil setelah Neo4j enrichment (Tahap 2)."""
    isi_id:        str
    jurnal_id:     str
    sub_judul:     str
    halaman:       int
    konten_chunk:  str    # teks chunk TARGET (murni)
    context_text:  str    # prev + target + next (untuk reranking & LLM prompt)
    judul_jurnal:  str
    doi:           str
    penulis:       str
    tanggal_rilis: str
    vector_score:  float
    rerank_score:  float = 0.0


@dataclass
class RAGResponse:
    """Respons akhir pipeline."""
    answer:          object          # Generator[str] untuk streaming
    sources:         List[Dict]      # referensi untuk ditampilkan di UI
    final_chunks:    List[EnrichedChunk]
    processing_time: float
    retrieval_time: float = 0.0      # <- baru
    enrichment_time: float = 0.0     # <- baru
    rerank_time: float = 0.0         # <- baru
    intent:          str = "knowledge"  # 'knowledge' | 'social'


############################################################
# MODELS LOADER (Singleton)
############################################################

class RAGModels:
    """
    Singleton — model lokal dimuat sekali saja.

    Embedding + Reranker → GPU  (VRAM kini bebas karena LLM ada di Groq API)
    LLM                  → Groq API  (openai/gpt-oss-120b, streaming SSE)

    GROQ_API_KEY dibaca dari environment variable saat inisialisasi.
    """

    _instance = None

    @classmethod
    def reset(cls):
        """Paksa re-inisialisasi singleton — dipanggil saat module reload."""
        cls._instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialize()
            cls._instance = instance
        return cls._instance

    def _initialize(self):
        log.info("Memulai pemuatan model RAG...")

        llm_mode = CONFIG.get("llm_mode", "groq")
        if llm_mode == "local":
            log.info(
                "Placement: embedding=cpu  reranker=cpu  nlp=cpu  llm=local(%s)",
                CONFIG.get("local_llm_path", "?"),
            )
        else:
            log.info(
                "Placement: embedding=%s  reranker=%s  nlp=%s  llm=groq-api(%s)",
                CONFIG["embedding_device"],
                CONFIG["reranker_device"],
                "cuda" if CONFIG["nlp_device"] >= 0 else "cpu",
                CONFIG["groq_model"],
            )

        # ── 1. Embedding model ────────────────────────────────────────────────
        log.info("[1/4] Embedding: %s → %s", CONFIG["embedding_model"], CONFIG["embedding_device"])
        _t = time.perf_counter()
        self.embedding_model = SentenceTransformer(
            CONFIG["embedding_model"],
            device=CONFIG["embedding_device"],
        )
        # Lock melindungi embedding_model dari akses GPU bersamaan.
        # RAG pipeline dan embedder PDF berbagi model yang sama —
        # keduanya berjalan di background thread terpisah dan harus
        # antri lewat lock ini sebelum memanggil .encode().
        self.embedding_lock = threading.Lock()
        log.info("[1/4] Embedding siap  (%.2fs)", time.perf_counter() - _t)

        # ── 2. Reranker → GPU ─────────────────────────────────────────────────
        log.info("[2/4] Reranker: %s → %s", CONFIG["reranker_model"], CONFIG["reranker_device"])
        _t = time.perf_counter()
        self.reranker = CrossEncoder(
            CONFIG["reranker_model"],
            device=CONFIG["reranker_device"],
        )
        log.info("[2/4] Reranker siap  (%.2fs)", time.perf_counter() - _t)

        self.llm_mode = "groq"
        log.info("[3/4] Groq API client → model=%s", CONFIG["groq_model"])
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY tidak ditemukan di environment. "
                "Set environment variable sebelum menjalankan server."
            )
        self.groq_client = Groq(api_key=api_key)
        self.local_llm       = None
        self.local_tokenizer = None
        log.info("[3/4] Groq client siap.")

        # ── 4. NLP — IndoBERT (ID) & BERT-NER (EN) ───────────────────────────
        log.info("[4/4] Memuat NLP: IndoBERT (ID) & BERT-NER (EN)...")
        _t = time.perf_counter()

        self.nlp_id_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["nlp_id_model"]
        )
        self.nlp_id_fillmask = hf_pipeline(
            "fill-mask",
            model=CONFIG["nlp_id_model"],
            tokenizer=CONFIG["nlp_id_model"],
            device=CONFIG["nlp_device"],
            top_k=5,  
        )

        self.nlp_en_pipeline = hf_pipeline(
            "ner",
            model=CONFIG["nlp_en_model"],
            tokenizer=CONFIG["nlp_en_model"],
            aggregation_strategy="simple",
            device=CONFIG["nlp_device"],
        )

        log.info("[4/4] NLP siap  (%.2fs)", time.perf_counter() - _t)

        log.info("✓ Semua model berhasil dimuat.")

    def get_embedding(self, text: str) -> List[float]:
        """
        Embed satu teks → vektor float (GPU, no_grad, thread-safe).

        Menggunakan embedding_lock agar tidak bertabrakan dengan
        embed_batch_safe() yang dipanggil embedder PDF di thread lain.
        """
        with self.embedding_lock:
            with torch.no_grad():
                return self.embedding_model.encode(
                    text, convert_to_tensor=False
                ).tolist()

    def embed_batch_safe(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch teks → list vektor float (GPU, no_grad, thread-safe).

        Dipakai oleh embedder PDF saat ingest dokumen baru.
        Berbagi embedding_lock dengan get_embedding() — keduanya
        tidak akan menyentuh GPU bersamaan meski berjalan di thread berbeda.

        Catatan: batch besar akan memegang lock lebih lama.
        RAG query yang datang saat lock dipegang akan menunggu
        sampai batch selesai — ini wajar dan by design.
        """
        with self.embedding_lock:
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=False,
                    show_progress_bar=False,  # nonaktifkan progress bar di server
                )
                return [e.tolist() for e in embeddings]

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        Cross-encoder scoring (query, teks) di GPU.
        Return: list float — skor lebih tinggi = lebih relevan.
        """
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        with torch.no_grad():
            scores = self.reranker.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)

    # ── Konstanta NLP ─────────────────────────────────────────────────────────

    _ID_STOPWORDS = {
        "yang", "dan", "di", "ke", "dari", "ini", "itu",
        "dengan", "untuk", "pada", "adalah", "ada", "atau",
        "juga", "oleh", "sebagai", "dalam", "tidak", "akan",
        "dapat", "bisa", "sudah", "telah", "lebih", "serta",
        "apakah", "apa", "bagaimana", "mengapa", "kenapa",
        "jelaskan", "sebutkan", "coba", "tolong", "mohon",
    }

    # Kosakata domain pertanian — TIDAK boleh dikoreksi meskipun OOV di BERT umum
    _DOMAIN_VOCAB = {
        "fusarium", "antraknosa", "nematoda", "aflatoksin", "alternaria",
        "pythium", "phytophthora", "rhizoctonia", "sclerotinia", "botrytis",
        "xanthomonas", "pseudomonas", "erwinia", "agrobacterium", "ralstonia",
        "tungro", "blas", "kresek", "hawar", "busuk", "layu", "bercak",
        "embun", "tepung", "karat", "virus", "bakteri", "jamur", "cendawan",
        "aphid", "thrips", "whitefly", "mealybug", "wereng", "penggerek",
        "ulat", "kutu", "tungau", "nematoda", "belalang", "lalat",
        # nama tanaman domain
        "kentang", "tomat", "cabai", "jagung", "padi", "kedelai", "singkong",
        "ubi", "terong", "bawang", "wortel", "kubis", "selada", "kangkung",
    }

    def correct_typo_mlm(self, text: str) -> str:
        """
        Koreksi typo pada query Bahasa Indonesia menggunakan IndoBERT MLM.

        Algoritma:
          1. Tokenisasi tiap kata dengan IndoBERT tokenizer
          2. Kata yang menghasilkan token [UNK] atau terpecah jadi ≥4 sub-kata
             dianggap berpotensi typo (OOV = out-of-vocabulary)
          3. Kata OOV yang BUKAN kosakata domain pertanian di-mask ([MASK])
          4. IndoBERT fill-mask memprediksi kandidat pengganti berdasarkan konteks
          5. Kandidat terbaik dipilih jika skornya ≥ threshold (0.15)
             dan lebih panjang dari 2 karakter (hindari prediksi noise)
          6. Hasil: query dengan kata typo sudah terkoreksi

        Catatan:
          - Kosakata domain pertanian (fusarium, antraknosa, dll) TIDAK dikoreksi
            karena memang OOV di BERT generik tapi valid secara domain
          - Threshold 0.15 cukup konservatif — hanya koreksi jika model yakin
          - Jika fill-mask gagal atau kata tidak ada kandidat baik → kata asli dipertahankan
        """
        words = text.split()
        corrected_words: list[str] = []
        any_corrected = False

        for word in words:
            word_lower = word.lower()

            # Kata domain → skip koreksi
            if word_lower in self._DOMAIN_VOCAB:
                corrected_words.append(word)
                continue

            # Cek apakah kata ini OOV di IndoBERT
            tokens = self.nlp_id_tokenizer.tokenize(word_lower)
            is_unk = "[UNK]" in tokens
            # Wordpiece memecah kata asing menjadi banyak sub-kata
            is_heavily_split = len(tokens) >= 4 and all(
                t.startswith("##") or len(t) <= 2 for t in tokens[1:]
            )

            if not (is_unk or is_heavily_split):
                # Kata dikenal dengan baik → pertahankan
                corrected_words.append(word)
                continue

            # Coba koreksi dengan fill-mask
            # Ganti kata ini dengan [MASK] dalam kalimat penuh untuk konteks
            masked_sentence = " ".join(
                "[MASK]" if w.lower() == word_lower else w
                for w in words
            )

            try:
                predictions = self.nlp_id_fillmask(masked_sentence)
                best = None
                for pred in predictions:
                    candidate = pred["token_str"].strip().lower()
                    score     = pred["score"]
                    # Filter: skor cukup tinggi, bukan noise, bukan sama persis
                    if (score >= 0.15
                            and len(candidate) > 2
                            and candidate != word_lower):
                        best = candidate
                        break

                if best:
                    log.debug(
                        "[MLM-Typo] '%s' → '%s' (score=%.3f)",
                        word, best, predictions[0]["score"],
                    )
                    corrected_words.append(best)
                    any_corrected = True
                else:
                    corrected_words.append(word)

            except Exception:
                log.warning("[MLM-Typo] fill-mask gagal untuk kata '%s'", word, exc_info=False)
                corrected_words.append(word)

        corrected_text = " ".join(corrected_words)
        if any_corrected:
            log.info("[MLM-Typo] Query terkoreksi: %r → %r", text, corrected_text)

        return corrected_text

    def extract_keywords_nlp(self, text: str, lang: str) -> str:
        """
        Ekstraksi keyword/entitas dari query menggunakan NLP:
          - lang='id' → IndoBERT: tokenisasi sub-kata, ambil token unik
                        non-stopword sebagai keyword tambahan.
          - lang='en' → BERT-NER: ambil entitas yang dikenali sebagai
                        keyword tambahan.

        Return: string keyword yang digabung ke query asli sebelum embedding,
                sehingga vektor lebih representatif terhadap entitas penting.
        """
        try:
            if lang == "id":
                tokens = self.nlp_id_tokenizer.tokenize(text)
                clean_tokens = [
                    t.replace("##", "").lower()
                    for t in tokens
                    if not t.startswith("[") and len(t.replace("##", "")) > 2
                ]
                keywords = [
                    t for t in dict.fromkeys(clean_tokens)
                    if t not in self._ID_STOPWORDS
                ]
                extra = " ".join(keywords[:10])
                log.debug("[NLP-ID] keywords: %s", extra)

            else:  # lang == 'en'
                ner_results = self.nlp_en_pipeline(text)
                keywords = list(dict.fromkeys(
                    entity["word"]
                    for entity in ner_results
                    if entity.get("score", 0) >= 0.7
                ))
                extra = " ".join(keywords[:10])
                log.debug("[NLP-EN] entities: %s", extra)

        except Exception:
            log.warning("[NLP] Ekstraksi keyword gagal, lanjut tanpa enrichment", exc_info=True)
            extra = ""

        return extra


############################################################
# TAHAP 1 — CHROMADB RETRIEVER
############################################################

class ChromaRetriever:
    """
    Wide retrieval dari ChromaDB collection 'konten_isi'.
    Metadata yang dikembalikan: isi_id (kunci ke Neo4j) + jurnal_id.
    """

    def __init__(self, persist_directory: str = CONFIG["chroma_path"]):
        log.info("ChromaDB: %s", persist_directory)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(CONFIG["chroma_collection"])
        log.info(
            "ChromaDB siap — '%s'  (%d dokumen)",
            CONFIG["chroma_collection"],
            self.collection.count(),
        )

    def retrieve(
        self,
        query_embedding: List[float],
        k: int = CONFIG["chroma_retrieval_k"],
    ) -> List[CandidateChunk]:
        """
        Cari k chunk paling mirip dengan query_embedding.
        Return: list CandidateChunk, urut dari yang paling dekat.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            log.exception("ChromaDB query gagal (k=%d)", k)
            return []

        candidates: List[CandidateChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return candidates

        for i, doc_id in enumerate(results["ids"][0]):
            meta = (results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0] else {})
            dist = (float(results["distances"][0][i])
                    if results["distances"] and results["distances"][0] else 1.0)

            candidates.append(CandidateChunk(
                isi_id=meta.get("isi_id", doc_id),
                jurnal_id=meta.get("jurnal_id", ""),
                konten_chunk=results["documents"][0][i],
                vector_score=dist,
            ))

        log.debug("ChromaDB: %d kandidat ditemukan (k=%d)", len(candidates), k)
        return candidates


############################################################
# TAHAP 2 — NEO4J ENRICHER
############################################################

class Neo4jEnricher:
    """
    ... (docstring tidak berubah)
    """

    def __init__(
        self,
        uri:      str = CONFIG["neo4j_uri"],
        user:     str = CONFIG["neo4j_user"],
        password: str = CONFIG["neo4j_password"],
        max_wait_seconds: int = 300,   # nunggu maksimal 5 menit sebelum menyerah
        retry_interval: int = 5,        # cek ulang tiap 5 detik
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        waited = 0
        while True:
            try:
                self.driver.verify_connectivity()
                log.info("Neo4j: %s (terhubung)", uri)
                break
            except Exception as e:
                if waited == 0:
                    print(f"\n⚠️  Neo4j belum aktif di {uri}")
                    print(f"   Silakan nyalakan Neo4j server sekarang.")
                print(f"   Menunggu koneksi Neo4j... ({waited}s / {max_wait_seconds}s)", end="\r")

                if waited >= max_wait_seconds:
                    raise ConnectionError(
                        f"Neo4j tidak dapat dihubungi setelah {max_wait_seconds}s di {uri}. "
                        f"Pastikan server aktif lalu jalankan ulang."
                    ) from e

                time.sleep(retry_interval)
                waited += retry_interval

    def close(self):
        self.driver.close()
    def enrich(
        self,
        candidates: List[CandidateChunk],
        context_window: int = CONFIG["context_window"],
    ) -> List[EnrichedChunk]:
        """
        Jalankan satu Cypher UNWIND untuk semua isi_id sekaligus.
        Kembalikan list EnrichedChunk dengan context_text berisi
        teks gabungan: [prev_chunks...] + target + [next_chunks...].
        """
        if not candidates:
            return []

        isi_ids  = [c.isi_id for c in candidates]
        cand_map = {c.isi_id: c for c in candidates}

        # Cypher: traverse mundur untuk prev, maju untuk next
        # Variabel {cw} diganti dengan nilai context_window
        cypher = (
            "UNWIND $isi_ids AS target_id "
            "MATCH (isi:Isi {id: target_id}) "
            "MATCH (j:Jurnal)-[:HAS_SECTION]->(isi) "

            # prev: node yang mengarah ke isi via NEXT (arah balik)
            "OPTIONAL MATCH (prev_isi:Isi)-[:NEXT*1..%(cw)d]->(isi) "
            "WITH isi, j, target_id, "
            "     collect(DISTINCT prev_isi.konten_chunk) AS prev_chunks "

            # next: node yang isi arahkan via NEXT
            "OPTIONAL MATCH (isi)-[:NEXT*1..%(cw)d]->(next_isi:Isi) "
            "WITH isi, j, target_id, prev_chunks, "
            "     collect(DISTINCT next_isi.konten_chunk) AS next_chunks "

            "RETURN "
            "  target_id        AS isi_id, "
            "  j.id             AS jurnal_id, "
            "  isi.sub_judul    AS sub_judul, "
            "  isi.halaman      AS halaman, "
            "  isi.konten_chunk AS konten_chunk, "
            "  j.judul          AS judul_jurnal, "
            "  j.doi            AS doi, "
            "  j.penulis        AS penulis, "
            "  j.tanggal_rilis  AS tanggal_rilis, "
            "  prev_chunks      AS prev_chunks, "
            "  next_chunks      AS next_chunks"
        ) % {"cw": context_window}

        enriched: List[EnrichedChunk] = []

        try:
            with self.driver.session() as session:
                for rec in session.run(cypher, isi_ids=isi_ids):
                    isi_id = rec["isi_id"]
                    cand   = cand_map.get(isi_id)
                    if cand is None:
                        continue

                    prev_list = [t for t in (rec["prev_chunks"] or []) if t]
                    next_list = [t for t in (rec["next_chunks"] or []) if t]
                    target    = rec["konten_chunk"] or ""

                    # Gabung: prev (urut maju) + target + next
                    context_text = " ".join([*prev_list, target, *next_list]).strip()

                    enriched.append(EnrichedChunk(
                        isi_id=isi_id,
                        jurnal_id=rec["jurnal_id"] or cand.jurnal_id,
                        sub_judul=rec["sub_judul"] or "Unknown",
                        halaman=int(rec["halaman"] or 0),
                        konten_chunk=target,
                        context_text=context_text,
                        judul_jurnal=rec["judul_jurnal"] or "Unknown",
                        doi=rec["doi"] or "",
                        penulis=rec["penulis"] or "Unknown",
                        tanggal_rilis=rec["tanggal_rilis"] or "Unknown",
                        vector_score=cand.vector_score,
                    ))

        except Exception:
            log.exception("Neo4j enrichment gagal (%d isi_id)", len(isi_ids))

        log.debug(
            "Neo4j enrich: %d/%d berhasil diperkaya",
            len(enriched), len(candidates),
        )
        return enriched


############################################################
# PIPELINE UTAMA — ROUTER + 2 JALUR TERPISAH
############################################################

class RAGPipeline:
    """
    Orkestrasi dua pipeline terpisah:

    ┌─────────────────────────────────────────────────────────┐
    │  process_query()  ←  router via _detect_query_intent()  │
    └───────────┬─────────────────────────┬───────────────────┘
                │ intent='knowledge'       │ intent='social'
                ▼                         ▼
    process_knowledge_query()    process_social_query()
      Tahap 1: ChromaDB            Prompt minimal → LLM
      Tahap 2: Neo4j enrich        temperature=0.8 (natural)
      Tahap 3: BGE reranking       max_new_tokens=128
      Tahap 4: Filtering
      Tahap 5: Memory inject (jika ada)
      Tahap 6: LLM (temp=0.2)

    Memory System:
      - get_memory(chat_id, query) → similarity search Q&A pairs dari ChromaDB 'chat_memory'
      - save_memory(chat_id, detail_id, question, answer) → simpan Q&A pair baru
      Keduanya dipanggil dari service/chats.py — save via _save_memory_entry (thread daemon).
    """

    def __init__(self):
        log.info("Menginisialisasi RAGPipeline...")
        self.models = RAGModels()
        self.chroma = ChromaRetriever()
        self.neo4j  = Neo4jEnricher()
        log.info("✓ RAGPipeline siap.")

    def close(self):
        self.neo4j.close()

    # ── Public API — Router ───────────────────────────────────────────────────

    def process_query(
        self,
        query:      str,
        chat_id:    int | None = None,
        stop_event: threading.Event = None,
        user_id:    int | None = None,
        base64_image: str | None = None, # <--- TAMBAHAN ARGUMEN
    ) -> RAGResponse:
        """
        Entry point utama. Deteksi intent lalu delegasikan ke pipeline
        yang sesuai: knowledge → process_knowledge_query,
                     social    → process_social_query.

        Sebelum deteksi intent, query Bahasa Indonesia dikoreksi typo-nya
        terlebih dahulu menggunakan IndoBERT MLM (fill-mask).

        chat_id (opsional): diteruskan ke kedua pipeline untuk membaca
        dan menyimpan memory. Social pipeline menggunakan memory untuk
        mengingat informasi personal (nama, preferensi, dll).

        user_id (opsional): id user yang sedang login. Digunakan untuk
        mengambil identitas (nama, dll.) dari ChromaDB collection
        'user_identity'. Identitas digabung ke blok memory — TIDAK
        diinjek langsung ke base prompt.
        """

        if base64_image:
            log.info("═" * 60)
            log.info("[Vision RAG] Langkah 1: Menganalisis gambar menggunakan Gemini...")
            image_description = self._analyze_image(base64_image)
            log.info(f"[Vision RAG] Hasil Ekstraksi: {image_description[:100]}...")
            
            # --- PERBAIKAN: CEGAH GROQ MENGANALISIS PESAN ERROR GEMINI ---
            if image_description.startswith("Gagal") or image_description.startswith("Gambar diterima"):
                def error_stream():
                    yield f"⚠️ **Sistem Vision Error:**\n\n{image_description}\n\n_Tips: Ini biasanya terjadi karena batas limit API gratis per menit. Silakan tunggu sekitar 1 menit lalu coba kirim ulang gambar Anda._"
                
                return RAGResponse(
                    answer=error_stream(),
                    sources=[],
                    final_chunks=[],
                    processing_time=0.0,
                    intent="vision_error"
                )
            # --------------------------------------------------------------
            
            # Jika sukses, gabungkan hasil analisa gambar dengan query asli
            if query.strip() and query.strip() != "Tolong jelaskan gambar tanaman ini.":
                enriched_query = f"Pengguna mengunggah gambar dengan hasil analisis visi dari pakar berikut:\n'{image_description}'\n\nBerdasarkan analisis visual tersebut, pengguna bertanya: '{query}'. Tolong berikan jawaban yang komprehensif."
            else:
                enriched_query = f"Pengguna mengunggah gambar dengan hasil analisis visi dari pakar berikut:\n'{image_description}'\n\nTolong jelaskan kondisi tanaman tersebut, kemungkinan penyebab, dan cara penanganannya."
            
            log.info("[Vision RAG] Langkah 2: Mengirim gabungan teks ke pipeline Knowledge Retrieval...")
            return self.process_knowledge_query(enriched_query, chat_id=chat_id, stop_event=stop_event, user_id=user_id)
        # ----------------------------------------------

        # Jika tidak ada gambar, lanjutkan seperti biasa
        # ── Koreksi typo via IndoBERT MLM (hanya untuk query Bahasa Indonesia) ─
        lang_pre = self._detect_language(query)
        if lang_pre == "id":
            query = self.models.correct_typo_mlm(query)

        intent = self._detect_query_intent(query)
        log.info("Intent terdeteksi: %s — query=%r", intent, query[:80])

        rag_mode = CONFIG.get("rag_mode", "improved")

        if intent == "social":
            return self.process_social_query(query, chat_id=chat_id, stop_event=stop_event, user_id=user_id)
        
        if rag_mode == "regular":
            log.info("[Router] Menggunakan Regular RAG pipeline")
            return self.process_regular_query(query, chat_id=chat_id, stop_event=stop_event, user_id=user_id)
    
        return self.process_knowledge_query(query, chat_id=chat_id, stop_event=stop_event, user_id=user_id)
    

    # ── Memory System ─────────────────────────────────────────────────────────

    def get_memory(self, chat_id: int, query: str, user_id: int | None = None) -> str | None:
        """
        Ambil hybrid memory dari ChromaDB collection 'chat_memory',
        dan gabungkan dengan identitas user dari collection 'user_identity'
        jika user_id disediakan.

        Tiga blok digabungkan (jika tersedia):
          0. Identitas user (identity_{user_id}) — dari collection terpisah.
             Berisi nama user dan informasi persisten lintas topic.

          1. Running summary (summary_{chat_id}) — konteks jangka panjang.
             Berisi topik-topik yang sudah dibahas dan ringkasan percakapan.

          2. Recent window (recent_{chat_id}_*) — N entry Q&A terbaru,
             diambil kronologis TANPA similarity search. Menjawab pertanyaan
             referensial temporal seperti "barusan", "tadi", "sebelumnya".

        query tidak dipakai untuk filtering — disertakan hanya untuk
        kompatibilitas signature dengan pemanggil di process_*_query.
        """
        try:
            collection = self.chroma.client.get_or_create_collection(
                CONFIG["memory_collection"]
            )

            # ── Blok 0: Identitas user dari collection 'user_identity' ────────
            identity: str = ""
            if user_id is not None:
                identity = self.get_identity(user_id) or ""

            # ── Blok 1: Running summary ───────────────────────────────────────
            summary: str = ""
            try:
                result = collection.get(
                    ids=[f"summary_{chat_id}"],
                    include=["documents"],
                )
                if result["ids"]:
                    summary = result["documents"][0]
                    log.info(
                        "[Memory] Summary ditemukan chat_id=%d  (%d char)",
                        chat_id, len(summary),
                    )
            except Exception:
                log.debug("[Memory] Belum ada summary untuk chat_id=%d", chat_id)

            # ── Blok 2: Recent window — N entry terbaru secara kronologis ─────
            # Filter by id prefix 'recent_{chat_id}_' — lebih reliable daripada
            # where filter karena tidak bergantung pada tipe data metadata di ChromaDB.
            recent_text: str = ""
            try:
                # Ambil semua entry di collection, lalu filter manual by id prefix
                # Ini menghindari masalah ChromaDB where filter dengan $and operator
                # dan inkonsistensi tipe int vs string pada metadata chat_id
                all_results = collection.get(include=["documents", "metadatas"])

                prefix = f"recent_{chat_id}_"
                matched_ids  = []
                matched_docs = []
                matched_meta = []

                for i, doc_id in enumerate(all_results["ids"]):
                    if doc_id.startswith(prefix):
                        matched_ids.append(doc_id)
                        matched_docs.append(all_results["documents"][i])
                        matched_meta.append(all_results["metadatas"][i])

                if matched_ids:
                    # Urutkan berdasarkan timestamp ascending (terlama ke terbaru)
                    entries = sorted(
                        zip(matched_docs, matched_meta),
                        key=lambda x: x[1].get("timestamp", 0),
                    )

                    # Ambil N terbaru sesuai config
                    n = CONFIG["memory_recent_window"]
                    entries = entries[-n:]

                    lines = [doc for doc, _ in entries]
                    recent_text = "\n\n".join(lines)
                    log.info(
                        "[Memory] Recent window: %d entry (dari %d total) chat_id=%d",
                        len(entries), len(matched_ids), chat_id,
                    )
                else:
                    log.debug(
                        "[Memory] Belum ada recent entries untuk chat_id=%d", chat_id
                    )

            except Exception:
                log.debug(
                    "[Memory] Gagal ambil recent entries untuk chat_id=%d", chat_id,
                    exc_info=True,
                )

            # ── Gabungkan tiga blok ───────────────────────────────────────────
            if not identity and not summary and not recent_text:
                log.debug("[Memory] Tidak ada memory untuk chat_id=%d", chat_id)
                return None

            parts = []
            if identity:
                parts.append(f"### IDENTITAS PENGGUNA ###\n{identity}")
            if summary:
                parts.append(f"### RINGKASAN SESI ###\n{summary}")
            if recent_text:
                parts.append(f"### PERCAKAPAN TERAKHIR ###\n{recent_text}")

            combined = "\n\n".join(parts)
            log.info(
                "[Memory] Hybrid memory siap — chat_id=%d  (%d char)",
                chat_id, len(combined),
            )
            return combined

        except Exception:
            log.warning(
                "[Memory] Gagal mengambil memory chat_id=%d", chat_id,
                exc_info=False,
            )
            return None

    def get_identity(self, user_id: int) -> str | None:
        """
        Ambil identitas user dari ChromaDB collection 'user_identity'.

        Identitas berisi informasi persisten tentang user seperti nama
        yang disimpan saat pertama kali chat. Berbeda dari chat_memory
        yang terikat per chat_id, identity terikat per user_id sehingga
        persisten lintas semua topic dan tidak ikut terhapus saat topic
        dihapus.

        Return: string teks identitas, atau None jika belum ada.
        """
        try:
            collection = self.chroma.client.get_or_create_collection(
                CONFIG["identity_collection"]
            )
            result = collection.get(
                ids=[f"identity_{user_id}"],
                include=["documents"],
            )
            if result["ids"]:
                identity_text = result["documents"][0]
                log.info(
                    "[Identity] Ditemukan user_id=%d  (%d char)",
                    user_id, len(identity_text),
                )
                return identity_text
            log.debug("[Identity] Belum ada identitas untuk user_id=%d", user_id)
            return None
        except Exception:
            log.warning(
                "[Identity] Gagal mengambil identitas user_id=%d", user_id,
                exc_info=False,
            )
            return None

    def save_identity(self, user_id: int, user_name: str) -> None:
        """
        Simpan atau perbarui identitas user di ChromaDB collection 'user_identity'.

        Dipanggil dari chats.py (_rag_worker) setiap kali chat diproses,
        sehingga jika nama user berubah di tabel users, identity di ChromaDB
        ikut diperbarui. Operasi upsert — aman dipanggil berulang kali.

        Format dokumen yang disimpan:
          "Nama pengguna: {user_name}"
        Format ini sengaja dibuat singkat dan mudah diparsing oleh LLM
        ketika dibaca sebagai bagian dari blok memory.

        user_id   : id integer dari tabel users (kunci lookup)
        user_name : nama lengkap user dari tabel users
        """
        if not user_name or not user_name.strip():
            log.debug("[Identity] user_name kosong — skip save user_id=%d", user_id)
            return
        try:
            collection = self.chroma.client.get_or_create_collection(
                CONFIG["identity_collection"]
            )
            identity_text = f"Nama pengguna: {user_name.strip()}"
            identity_embedding = self.models.get_embedding(identity_text)
            collection.upsert(
                ids=[f"identity_{user_id}"],
                documents=[identity_text],
                embeddings=[identity_embedding],
                metadatas=[{
                    "user_id":    user_id,
                    "user_name":  user_name.strip(),
                    "updated_at": int(time.time()),
                }],
            )
            log.info(
                "[Identity] Disimpan → user_id=%d  user_name=%r",
                user_id, user_name,
            )
        except Exception:
            log.exception(
                "[Identity] Gagal menyimpan identitas user_id=%d", user_id
            )

    def save_memory(self, chat_id: int, detail_id: int, question: str, answer: str) -> None:
        """
        Simpan memory hybrid ke ChromaDB collection 'chat_memory'.

        Dua operasi dijalankan dalam satu pemanggilan:

          1. Update running summary (id tetap 'summary_{chat_id}').
             Summary lama + Q&A baru dirangkum ulang oleh LLM.
             Instruksi prioritas memaksa topik utama tidak pernah dihapus
             meski terjadi kompresi.

          2. Simpan entry episodik baru (id 'recent_{chat_id}_{detail_id}').
             Format teks: "User: ...\nragna: ..."
             Metadata: chat_id, detail_id, timestamp, type='recent'
             Dipakai oleh get_memory() sebagai recent window kronologis.

        Dipanggil oleh chats.py setelah response berhasil di-commit ke DB.
        Identitas user (nama dll.) TIDAK disimpan di sini — gunakan
        save_identity() secara terpisah di collection 'user_identity'.
        """
        if not answer or not answer.strip():
            log.warning(
                "[Memory] Answer kosong — skip save chat_id=%d detail_id=%d",
                chat_id, detail_id,
            )
            return

        try:
            collection = self.chroma.client.get_or_create_collection(
                CONFIG["memory_collection"]
            )

            # ══════════════════════════════════════════════════════════════════
            # BAGIAN 1 — Update running summary
            # ══════════════════════════════════════════════════════════════════

            # ── Ambil summary lama jika ada ───────────────────────────────────
            previous_summary: str = ""
            try:
                existing = collection.get(
                    ids=[f"summary_{chat_id}"],
                    include=["documents"],
                )
                if existing["ids"]:
                    previous_summary = existing["documents"][0]
            except Exception:
                pass  # Belum ada summary — mulai dari kosong

            max_words = CONFIG["memory_summary_max_words"]

            # Truncate agar total prompt summarizer tidak meledak
            _max_answer_chars  = CONFIG["memory_summary_max_tokens"] * 3   # ~1536 char untuk 512 token
            _max_summary_chars = CONFIG["memory_summary_max_tokens"] * 2   # ~1024 char
            _max_question_chars = 400

            answer_trunc   = answer.strip()[:_max_answer_chars]
            question_trunc = question.strip()[:_max_question_chars]
            prev_trunc     = previous_summary[:_max_summary_chars] if previous_summary else ""

            if prev_trunc:
                summary_prompt = PROMPTS["memory_summary_update"].format(
                    max_words=max_words,
                    previous_summary=prev_trunc,
                    question=question_trunc,
                    answer=answer_trunc,
                )
            else:
                summary_prompt = PROMPTS["memory_summary_new"].format(
                    max_words=max_words,
                    question=question_trunc,
                    answer=answer_trunc,
                )

            # ── Panggil LLM untuk summarization ──────────────────────────────
            log.info(
                "[Memory] Merangkum summary baru — chat_id=%d  detail_id=%d  "
                "prev_summary=%d char",
                chat_id, detail_id, len(previous_summary),
            )
            # Summarizer ikut mode LLM yang aktif (groq atau local)
            if self.models.llm_mode == "groq":
                # ── Groq API ──────────────────────────────────────────────────────
                summary_response = self.models.groq_client.chat.completions.create(
                    model=CONFIG["memory_summary_model"],
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=CONFIG["memory_summary_max_tokens"],
                    temperature=0.3,
                )
                new_summary = summary_response.choices[0].message.content.strip()
            else:
                # ── Local LLM ─────────────────────────────────────────────────────
                # Kumpulkan seluruh token dari generator (summarizer tidak perlu streaming)
                tokenizer = self.models.local_tokenizer
                model     = self.models.local_llm

                inputs = tokenizer(
                    summary_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).input_ids.to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        inputs,
                        max_new_tokens=CONFIG["memory_summary_max_tokens"],
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Potong token input — ambil hanya bagian yang di-generate
                generated = output_ids[0][inputs.shape[-1]:]
                new_summary = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # ── Upsert summary ke ChromaDB (overwrite entry lama) ─────────────
            summary_embedding = self.models.get_embedding(new_summary)
            collection.upsert(
                ids=[f"summary_{chat_id}"],
                documents=[new_summary],
                embeddings=[summary_embedding],
                metadatas=[{
                    "type":           "summary",
                    "chat_id":        chat_id,
                    "last_detail_id": detail_id,
                }],
            )
            log.info(
                "[Memory] Summary diperbarui → chat_id=%d  detail_id=%d  (%d char)",
                chat_id, detail_id, len(new_summary),
            )

            # ══════════════════════════════════════════════════════════════════
            # BAGIAN 2 — Simpan entry episodik (recent window)
            # ══════════════════════════════════════════════════════════════════
            recent_doc = (
                f"User: {question.strip()}\n"
                f"ragna: {answer.strip()}"
            )
            recent_embedding = self.models.get_embedding(recent_doc)
            collection.upsert(
                ids=[f"recent_{chat_id}_{detail_id}"],
                documents=[recent_doc],
                embeddings=[recent_embedding],
                metadatas=[{
                    "type":      "recent",
                    "chat_id":   chat_id,
                    "detail_id": detail_id,
                    "timestamp": int(time.time()),
                }],
            )
            log.info(
                "[Memory] Recent entry disimpan → chat_id=%d  detail_id=%d",
                chat_id, detail_id,
            )

        except Exception:
            log.exception(
                "[Memory] Gagal update memory chat_id=%d detail_id=%d",
                chat_id, detail_id,

            )

    # ── Social Pipeline ───────────────────────────────────────────────────────
    def process_social_query(
        self,
        query:      str,
        chat_id:    int | None = None,
        stop_event: threading.Event = None,
        user_id:    int | None = None,
    ) -> RAGResponse:
        t_start = time.perf_counter()
        lang    = self._detect_language(query)
        tier    = self._get_model_tier()

        # ── Ambil memory + identitas user jika tersedia ───────────────────────
        memory_text: str | None = None
        if chat_id is not None:
            memory_text = self.get_memory(chat_id, query, user_id=user_id)
            if memory_text:
                log.info("[Social] Memory ditemukan (%d char)", len(memory_text))
            else:
                log.debug("[Social] Belum ada memory untuk chat_id=%d", chat_id)

        # ── Pilih prompt berdasarkan tier model ───────────────────────────────
        # large  → few-shot penuh (model besar mampu memisahkan contoh dari instruksi)
        # medium / small → prompt ringkas (model kecil cenderung mereproduksi contoh)
        use_compact = tier in ("small", "medium")

        if lang == "id":
            memory_section = (
                PROMPTS["social_memory_block_id"].format(memory=memory_text)
                if memory_text else ""
            )
            prompt_key = "social_system_id_local" if use_compact else "social_system_id"
            system_msg = PROMPTS[prompt_key].format(memory_section=memory_section)
        else:
            memory_section = (
                PROMPTS["social_memory_block_en"].format(memory=memory_text)
                if memory_text else ""
            )
            prompt_key = "social_system_en_local" if use_compact else "social_system_en"
            system_msg = PROMPTS[prompt_key].format(memory_section=memory_section)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": query},
        ]

        log.info(
            "[Social] tier=%s  model=%s  prompt=%s  lang=%s  memory=%s  user_id=%s  query=%r",
            tier,
            CONFIG.get("groq_model", "?"),
            prompt_key,
            lang,
            "ya" if memory_text else "tidak",
            user_id or "-",
            query[:60],
        )

        answer_gen = self._generate_stream(
            messages,
            stop_event=stop_event,
            temperature=CONFIG["social_temperature"],
            top_p=CONFIG["social_top_p"],
            max_new_tokens=CONFIG["social_max_new_tokens"],
        )

        elapsed = time.perf_counter() - t_start
        return RAGResponse(
            answer=answer_gen,
            sources=[],
            final_chunks=[],
            processing_time=elapsed,
            intent="social",
        )

    # ── Knowledge Pipeline ────────────────────────────────────────────────────

    def process_knowledge_query(
        self,
        query:      str,
        chat_id:    int | None = None,
        stop_event: threading.Event = None,
        user_id:    int | None = None,
    ) -> RAGResponse:
        """
        Pipeline knowledge — WITH retrieval (6 tahap).
        Tahap 1 → ChromaDB wide retrieval
        Tahap 2 → Neo4j context enrichment
        Tahap 3 → BGE reranking (GPU)
        Tahap 4 → Filtering & diversifikasi sumber
        Tahap 5 → Memory inject (identity + chat_memory dari ChromaDB)
        Tahap 6 → LLM streaming generation (Groq API, temp=0.2)

        chat_id digunakan di Tahap 5 untuk mengambil memory summary.
        Jika None (misal dari simple_retrieval), memory dilewati.

        user_id (opsional): digunakan di Tahap 5 agar get_memory() dapat
        menyertakan identitas user (nama dll.) dari collection 'user_identity'.
        Nama user TIDAK diinjek ke base prompt — hanya lewat blok memory.
        """
        t_start = time.perf_counter()
        log.info("═" * 60)
        log.info("[Knowledge] Query: %r  chat_id=%s", query[:120], chat_id)

        # ── Deteksi bahasa & NLP keyword enrichment ───────────────────────────
        lang = self._detect_language(query)
        log.info("[Knowledge] Bahasa terdeteksi: %s", lang)

        nlp_keywords = self.models.extract_keywords_nlp(query, lang)
        enriched_query = f"{query} {nlp_keywords}".strip() if nlp_keywords else query
        log.debug("[Knowledge] Query diperkaya: %r", enriched_query[:200])

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 1 — ChromaDB Wide Retrieval
        # ══════════════════════════════════════════════════════════════════════
        k = CONFIG["chroma_retrieval_k"]
        log.info("[Tahap 1] ChromaDB retrieval (k=%d)...", k)
        t1_start = time.perf_counter()

        query_emb  = self.models.get_embedding(enriched_query)
        candidates = self.chroma.retrieve(query_emb, k=k)

        retrieval_elapsed = time.perf_counter() - t1_start
        log.info("[Tahap 1] %d kandidat ditemukan  (%.3fs)", len(candidates), retrieval_elapsed)  # <-- FIX #1

        if not candidates:
            log.warning("[Tahap 1] Tidak ada kandidat — pipeline berhenti.")
            return RAGResponse(
                answer="Maaf, tidak menemukan informasi relevan di database.",
                sources=[],
                final_chunks=[],
                processing_time=time.perf_counter() - t_start,
                retrieval_time=retrieval_elapsed,           # <-- FIX (tambahan, early exit)
                intent="knowledge",
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 2 — Neo4j Context Enrichment
        # ══════════════════════════════════════════════════════════════════════
        log.info(
            "[Tahap 2] Neo4j enrichment (%d kandidat, window=±%d)...",
            len(candidates), CONFIG["context_window"],
        )
        t2_start = time.perf_counter()

        enriched = self.neo4j.enrich(candidates, CONFIG["context_window"])

        enrichment_elapsed = time.perf_counter() - t2_start
        log.info("[Tahap 2] %d chunk diperkaya  (%.3fs)", len(enriched), enrichment_elapsed)  # <-- FIX #2

        if not enriched:
            log.warning("[Tahap 2] Enrichment kosong — pipeline berhenti.")
            return RAGResponse(
                answer="Maaf, gagal mengambil konteks dari graph database.",
                sources=[],
                final_chunks=[],
                processing_time=time.perf_counter() - t_start,
                retrieval_time=retrieval_elapsed,           # <-- FIX (tambahan, early exit)
                enrichment_time=enrichment_elapsed,          # <-- FIX (tambahan, early exit)
                intent="knowledge",
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 3 — BGE Reranking
        # ══════════════════════════════════════════════════════════════════════
        reranked_k = CONFIG["reranked_k"]
        log.info(
            "[Tahap 3] BGE reranking (%d → top %d) @ GPU...",
            len(enriched), reranked_k,
        )

        t3_start = time.perf_counter()

        scores = self.models.rerank(query, [c.context_text for c in enriched])

        for i, score in enumerate(scores):
            if i < len(enriched):
                enriched[i].rerank_score = float(score)

        enriched.sort(key=lambda x: x.rerank_score, reverse=True)
        top_chunks = enriched[:reranked_k]

        rerank_elapsed = time.perf_counter() - t3_start
        log.info(                                                                            # <-- FIX #3
            "[Tahap 3] Top %d dipilih  (%.3fs)  skor: min=%.4f  max=%.4f",
            len(top_chunks), rerank_elapsed,
            min(c.rerank_score for c in top_chunks),
            max(c.rerank_score for c in top_chunks),
        )


        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 4 — Filtering & Diversifikasi Sumber
        #
        # Strategi dua lapis:
        #   a) max_chunks_per_jurnal — cegah satu jurnal mendominasi konteks
        #   b) final_context_k       — batasi total chunk ke LLM agar prompt
        #                              tidak melebihi context window
        # ══════════════════════════════════════════════════════════════════════
        max_per_j = CONFIG["max_chunks_per_jurnal"]
        final_k   = CONFIG["final_context_k"]
        log.info(
            "[Tahap 4] Filtering: max %d/jurnal → ambil top %d...",
            max_per_j, final_k,
        )

        jurnal_count: Dict[str, int] = {}
        final_chunks: List[EnrichedChunk] = []

        for chunk in top_chunks:
            jid = chunk.jurnal_id
            if jurnal_count.get(jid, 0) < max_per_j:
                final_chunks.append(chunk)
                jurnal_count[jid] = jurnal_count.get(jid, 0) + 1
            if len(final_chunks) >= final_k:
                break

        log.info(
            "[Tahap 4] Final: %d chunk dari %d jurnal",
            len(final_chunks), len(jurnal_count),
        )

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 5 — Memory Inject
        #
        # Ambil hybrid memory dari ChromaDB:
        #   - Identitas user (collection 'user_identity') via user_id
        #   - Running summary + recent window (collection 'chat_memory') via chat_id
        # Ketiganya digabung oleh get_memory() menjadi satu blok yang diinjek
        # ke system prompt. Nama user TIDAK ada di base prompt — hanya di sini.
        # ══════════════════════════════════════════════════════════════════════
        memory_text: str | None = None
        if chat_id is not None:
            log.info("[Tahap 5] Mengambil memory untuk chat_id=%d  user_id=%s...", chat_id, user_id)
            memory_text = self.get_memory(chat_id, query, user_id=user_id)
            if memory_text:
                log.info(
                    "[Tahap 5] Memory ditemukan  (%d char)", len(memory_text)
                )
            else:
                log.info("[Tahap 5] Belum ada memory — pertanyaan pertama atau belum ada entry relevan.")
        else:
            log.debug("[Tahap 5] chat_id=None — memory dilewati.")

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 6 — LLM Generation (Groq API, streaming)
        # ══════════════════════════════════════════════════════════════════════
        messages   = self._build_messages(query, final_chunks, lang=lang, memory=memory_text)
        answer_gen = self._generate_stream(
            messages,
            stop_event=stop_event,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            max_new_tokens=CONFIG["max_new_tokens"],
        )

        # ── Sumber referensi untuk UI ─────────────────────────────────────────
        sources = [
            {
                "sub_judul":    c.sub_judul,
                "jurnal":       c.judul_jurnal,
                "penulis":      c.penulis,
                "tahun":        c.tanggal_rilis,
                "doi":          c.doi or "-",
                "halaman":      c.halaman,
                "rerank_score": f"{c.rerank_score:.4f}",
                "vector_score": f"{c.vector_score:.4f}",
            }
            for c in final_chunks
        ]

        elapsed = time.perf_counter() - t_start
        log.info(
            "[Knowledge] Pipeline selesai — %.3fs  |  chunks=%d  sumber=%d  memory=%s",
            elapsed, len(final_chunks), len(sources),
            "ya" if memory_text else "tidak",
        )
        log.info("═" * 60)

        return RAGResponse(
            answer=answer_gen,
            sources=sources,
            final_chunks=top_chunks,
            processing_time=elapsed,
            retrieval_time=retrieval_elapsed,
            enrichment_time=enrichment_elapsed,
            rerank_time=rerank_elapsed,
            intent="knowledge",
        )
    
    def process_vision_query(
        self,
        query: str,
        base64_image: str,
        chat_id: int | None = None,
        user_id: int | None = None,
        stop_event: threading.Event = None,
    ) -> RAGResponse:
        t_start = time.perf_counter()
        log.info("═" * 60)
        log.info("[Vision] Memproses gambar dengan Gemini 3 Flash Preview...")

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY tidak ditemukan di file .env")
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')

        try:
            image_data = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(image_data))
        except Exception as e:
            log.error(f"[Vision] Gagal memuat gambar: {e}")
            raise ValueError("Data gambar tidak valid atau korup.")

        prompt = f"Sebagai pakar pertanian (ragna), tolong analisis gambar ini secara detail.\n\nKonteks/Pertanyaan pengguna: {query}"

        def stream_generator():
            try:
                response = model.generate_content([prompt, img], stream=True)
                for chunk in response:
                    if stop_event and stop_event.is_set():
                        log.info("[Gemini] Streaming dihentikan oleh user.")
                        break
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                log.error(f"[Gemini] Error saat streaming API: {e}")
                yield f"\n\n[Sistem] Maaf, terjadi kesalahan dari server Gemini. Detail: {e}"

        elapsed = time.perf_counter() - t_start
        log.info("[Vision] Gambar berhasil dikirim ke Gemini (%.3fs)", elapsed)
        log.info("═" * 60)

        return RAGResponse(
            answer=stream_generator(),
            sources=[],       
            final_chunks=[],
            processing_time=elapsed,
            intent="vision",
        )

    def process_regular_query(
        self,
        query:      str,
        chat_id:    int | None = None,
        stop_event: threading.Event = None,
        user_id:    int | None = None,
    ) -> RAGResponse:
        """
        Regular RAG Pipeline:
        Tahap 1: ChromaDB similarity search di raw collection
        Tahap 2: BGE reranking
        Tahap 3: Memory inject (opsional)
        Tahap 4: LLM generation
        
        Perbedaan dengan Improved RAG:
        - Tidak ada Neo4j context enrichment
        - Tidak ada window prev/next chunks
        - Tidak ada max_chunks_per_jurnal filtering
        - Retrieval langsung dari raw_chunk tanpa metadata jurnal kompleks
        """
        t_start = time.perf_counter()
        log.info("═" * 60)
        log.info("[RegularRAG] Query: %r  chat_id=%s", query[:120], chat_id)

        # ── Deteksi bahasa & NLP keyword enrichment ───────────────────────────
        lang = self._detect_language(query)
        nlp_keywords = self.models.extract_keywords_nlp(query, lang)
        enriched_query = f"{query} {nlp_keywords}".strip() if nlp_keywords else query

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 1 — ChromaDB Retrieval (RAW COLLECTION)
        # ══════════════════════════════════════════════════════════════════════
        k = CONFIG.get("regular_retrieval_k", 6)
        log.info("[RegularRAG Tahap 1] ChromaDB retrieval (k=%d) dari raw_collection...", k)
        t1 = time.perf_counter()

        query_emb = self.models.get_embedding(enriched_query)

        raw_collection = self.chroma.client.get_collection(CONFIG["raw_collection"])
        results = raw_collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        raw_chunks = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 1.0
                raw_chunks.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": meta,
                    "vector_score": dist,
                })

        raw_retrieval_elapsed = time.perf_counter() - t1                                       # <-- FIX #4a
        log.info("[RegularRAG Tahap 1] %d chunk ditemukan (%.3fs)", len(raw_chunks), raw_retrieval_elapsed)

        if not raw_chunks:
            return RAGResponse(
                answer="Maaf, tidak menemukan informasi relevan di database.",
                sources=[], final_chunks=[], processing_time=time.perf_counter() - t_start,
                retrieval_time=raw_retrieval_elapsed,                                           # <-- FIX #4b
                intent="knowledge",
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 2 — Reranking (BGE Cross-Encoder)
        # ══════════════════════════════════════════════════════════════════════
        reranked_k = CONFIG.get("regular_reranked_k", 3)
        log.info("[RegularRAG Tahap 2] BGE reranking (%d → top %d)...", len(raw_chunks), reranked_k)
        t2 = time.perf_counter()

        chunk_texts = [c["text"] for c in raw_chunks]
        scores = self.models.rerank(query, chunk_texts)

        for i, score in enumerate(scores):
            raw_chunks[i]["rerank_score"] = float(score)

        raw_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_chunks = raw_chunks[:reranked_k]

        raw_rerank_elapsed = time.perf_counter() - t2                                           # <-- FIX #4c
        log.info("[RegularRAG Tahap 2] Top %d dipilih (%.3fs)", len(top_chunks), raw_rerank_elapsed)

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 3 — Memory Inject
        # ══════════════════════════════════════════════════════════════════════
        memory_text: str | None = None
        if chat_id is not None:
            memory_text = self.get_memory(chat_id, query, user_id=user_id)

        # ══════════════════════════════════════════════════════════════════════
        # TAHAP 4 — LLM Generation
        # ══════════════════════════════════════════════════════════════════════
        messages = self._build_regular_messages(query, top_chunks, lang=lang, memory=memory_text)
        answer_gen = self._generate_stream(
            messages,
            stop_event=stop_event,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            max_new_tokens=CONFIG["max_new_tokens"],
        )

        sources = [
            {
                "chunk_text": c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
                "rerank_score": f"{c.get('rerank_score', 0):.4f}",
                "vector_score": f"{c['vector_score']:.4f}",
                "metadata": c.get("metadata", {}),
            }
            for c in top_chunks
        ]

        elapsed = time.perf_counter() - t_start
        log.info("[RegularRAG] Pipeline selesai — %.3fs  |  chunks=%d", elapsed, len(top_chunks))
        log.info("═" * 60)

        return RAGResponse(
            answer=answer_gen,
            sources=sources,
            final_chunks=top_chunks,
            processing_time=elapsed,
            retrieval_time=raw_retrieval_elapsed,        # <-- FIX #4d
            enrichment_time=0.0,                          # <-- FIX #4e (Raw memang tidak enrich)
            rerank_time=raw_rerank_elapsed,               # <-- FIX #4f
            intent="knowledge",
        )
    
    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _detect_query_intent(text: str) -> str:
        """
        Routing intent: 'knowledge' → RAG pipeline, 'social' → social pipeline.

        Prioritas pengecekan:
          1. Kata/frasa sosial ringan          → 'social'
          2. Tanda tanya (?)                   → 'knowledge'
          3. Kata tanya / perintah informatif  → 'knowledge'
          4. Default                           → 'social'
        """
        normalized = text.lower().strip()
        words      = set(normalized.split())

        # ── 1. Social keywords (prioritas tertinggi) ──────────────────────────
        SOCIAL_PHRASES = {
            "apa kabar", "apakabar", "terima kasih", "terimakasih",
            "sampai jumpa", "selamat tinggal", "thank you",
        }
        SOCIAL_WORDS = {
            "hai", "halo", "hello", "hi", "hey",
            "kabar", "makasih", "thanks",
            "maaf", "sorry", "permisi",
            "dadah", "bye",
            "oke", "ok", "baik", "siap", "sip",
            "namaku", "ingat", "siapa aku"
        }
        for phrase in SOCIAL_PHRASES:
            if phrase in normalized:
                log.debug("[Intent] social — frasa: %r", phrase)
                return "social"
        if words & SOCIAL_WORDS:
            log.debug("[Intent] social — kata sosial terdeteksi")
            return "social"

        # ── 2. Tanda tanya eksplisit ──────────────────────────────────────────
        if "?" in text:
            log.debug("[Intent] knowledge — tanda tanya")
            return "knowledge"

        # ── 3. Kata tanya / perintah informatif ──────────────────────────────
        KNOWLEDGE_PHRASES = {
            # frasa Indonesia umum
            "apa itu", "yang mana", "di mana",
            # perintah dengan awalan "coba"
            "coba ranking", "coba urutkan", "coba sebutkan", "coba jelaskan",
            "coba bandingkan", "coba ceritakan", "coba buat", "coba berikan",
            "coba tampilkan", "coba tunjukkan",
            # perintah dengan awalan "tolong"
            "tolong jelaskan", "tolong sebutkan", "tolong ranking",
            "tolong urutkan", "tolong buat", "tolong berikan", "tolong ceritakan",
            # perintah dengan awalan "bisa"
            "bisa jelaskan", "bisa sebutkan", "bisa ranking", "bisa urutkan",
            # frasa urutan/perbandingan
            "dari yang", "mulai dari", "urutan dari",
            "dari terbanyak", "dari terbesar", "dari tertinggi",
            "sampai yang sedikit", "sampai yang kecil", "sampai yang rendah",
        }
        KNOWLEDGE_WORDS = {
            # kata tanya Indonesia
            "apa", "apakah", "bagaimana", "mengapa", "kenapa",
            "siapa", "kapan", "dimana", "berapa", "seberapa", "manakah",
            # perintah informatif langsung
            "jelaskan", "sebutkan", "ceritakan", "gambarkan",
            "deskripsikan", "definisikan", "definisi", "contoh", "contohkan",
            "bandingkan", "bedakan", "perbedaan", "persamaan",
            "cara", "langkah", "proses", "prosedur", "metode",
            "penyebab", "akibat", "dampak", "gejala", "tanda",
            "pengertian", "maksud", "artinya", "fungsi", "manfaat",
            "ciri", "karakteristik", "jenis", "macam", "klasifikasi",
            "penanganan", "pengobatan", "pengendalian", "pencegahan",
            # perintah ranking/urutan — sering tanpa tanda tanya
            "ranking", "rangking", "urutan", "urutkan",
            "peringkat", "daftar", "susun", "susunkan",
            "terbanyak", "tersedikit", "terbesar", "terkecil",
            "tertinggi", "terendah", "terluas",
            # awalan perintah umum
            "buatkan", "berikan", "tampilkan", "tunjukkan",
            "rekomendasikan", "rekomendasi",
            # domain pertanian/hama/penyakit — query domain = knowledge
            "hama", "penyakit", "patogen", "serangan", "infeksi",
            "tanaman", "tumbuhan", "pertanian", "agronomi", "pestisida",
            "pupuk", "lahan", "sawah", "kebun", "panen", "benih", "bibit",
            # kata tanya Inggris
            "what", "how", "why", "when", "where", "who", "which",
            "explain", "describe", "list", "define", "compare", "rank",
            "causes", "symptoms", "treatment", "control", "prevention",
            "give", "show", "recommend", "provide",
        }
        for phrase in KNOWLEDGE_PHRASES:
            if phrase in normalized:
                log.debug("[Intent] knowledge — frasa: %r", phrase)
                return "knowledge"
        if words & KNOWLEDGE_WORDS:
            log.debug("[Intent] knowledge — kata tanya terdeteksi")
            return "knowledge"

        log.debug("[Intent] social — tidak ada indikator knowledge")
        return "social"

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Deteksi bahasa query.
        Default Indonesia — return 'en' hanya jika ada ≥2 marker Inggris.
        """
        en_markers = {
            "what", "how", "why", "when", "where", "who", "which",
            "explain", "describe", "tell", "list", "give", "show",
            "define", "compare", "is", "are", "does", "do", "can",
            "could", "the", "of", "in", "and", "or", "with", "for",
            "about", "symptoms", "disease", "plant", "fungus",
            "bacteria", "treatment", "control",
        }
        words    = set(text.lower().split())
        en_score = len(words & en_markers)
        return "en" if en_score >= 2 else "id"

    @staticmethod
    def _get_model_tier() -> str:
        model = CONFIG.get("groq_model", "").lower()

        # ── Deteksi ukuran dari nama model ────────────────────────────────────
        # Pola: angka sebelum 'b' (misal "70b", "3b", "8b", "32b")
        import re
        matches = re.findall(r"(\d+)b", model)
        if matches:
            size = max(int(m) for m in matches)
            if size <= 4:
                return "small"
            if size <= 40:
                return "medium"
            return "large"

        # Fallback keyword-based
        if any(k in model for k in ("70b", "72b", "8x22b", "mixtral")):
            return "large"
        if any(k in model for k in ("32b",)):
            return "large"
        if any(k in model for k in ("7b", "8b", "12b", "13b")):
            return "medium"
        if any(k in model for k in ("3b", "1b")):
            return "small"

        return "large"  # default ke large jika tidak dikenali

    def _build_messages(
        self,
        query:  str,
        chunks: List[EnrichedChunk],
        lang:   str = None,
        memory: str | None = None,
    ) -> List[Dict]:
        tier      = self._get_model_tier()
        safe_budget = GROQ_MODEL_SAFE_TOKEN_BUDGET.get(CONFIG["groq_model"], 4_800)
        _usable = safe_budget - FIXED_OVERHEAD_TOKENS
        max_chars = min(int(_usable * 0.30 * 4), CONFIG["context_max_chars"])

        # Kurangi context window untuk model kecil/medium
        if tier == "small":
            max_chars = min(max_chars, 3_000)   # ~750 token untuk context
        elif tier == "medium":
            max_chars = min(max_chars, 6_000)   # ~1500 token untuk context

        log.info(
            "[BuildMessages] tier=%s  context_max_chars(CONFIG)=%d  max_chars(efektif)=%d",
            tier, CONFIG["context_max_chars"], max_chars,
        )
        # large → pakai max_chars penuh dari CONFIG

        context_parts: List[str] = []
        used_chars = 0

        for i, c in enumerate(chunks, 1):
            part = f"[{i}] {c.sub_judul}\n{c.context_text}"
            if used_chars + len(part) > max_chars:
                remaining = max_chars - used_chars
                if remaining > 200:
                    context_parts.append(part[:remaining] + "…")
                break
            context_parts.append(part)
            used_chars += len(part)

        context_str = "\n\n".join(context_parts)

        source_lines = [
            f"[{i}] {c.judul_jurnal} — {c.penulis} ({c.tanggal_rilis})"
            + (f"  DOI: {c.doi}" if c.doi else "")
            + f"  hal. {c.halaman}"
            for i, c in enumerate(chunks, 1)
        ]
        source_str = "\n".join(source_lines)

        lang = lang if lang is not None else self._detect_language(query)

        if lang == "id":
            _max_memory_chars = 1_200  # ~300 token, cukup untuk context singkat
            if memory and len(memory) > _max_memory_chars:
                memory = memory[:_max_memory_chars] + "…"
                log.debug("[BuildMessages] Memory dipotong ke %d char", _max_memory_chars)
            memory_section = (
                PROMPTS["knowledge_memory_block_id"].format(memory=memory)
                if memory else ""
            )
            # Pilih prompt key berdasarkan tier model
            prompt_key = {
                "small":  "knowledge_system_id_local",   # prompt ringkas
                "medium": "knowledge_system_id_local",   # sama dengan small — padat
                "large":  "knowledge_system_id",          # prompt penuh + few-shot
            }[tier]
            system_content = PROMPTS[prompt_key].format(
                memory_section=memory_section,
                context_str=context_str,
                source_str=source_str,
            )
            question_label = "Pertanyaan"
        else:
            memory_section = (
                PROMPTS["knowledge_memory_block_en"].format(memory=memory)
                if memory else ""
            )
            prompt_key = {
                "small":  "knowledge_system_en_local",
                "medium": "knowledge_system_en_local",
                "large":  "knowledge_system_en",
            }[tier]
            system_content = PROMPTS[prompt_key].format(
                memory_section=memory_section,
                context_str=context_str,
                source_str=source_str,
            )
            question_label = "Question"

        log.debug(
            "[BuildMessages] tier=%s  model=%s  prompt=%s  lang=%s  context=%d char  memory=%s",
            tier,
            CONFIG.get("groq_model", "?"),
            prompt_key,
            lang,
            len(context_str),
            "ya" if memory else "tidak",
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": f"{question_label}: {query}"},
        ]

    def _build_regular_messages(
        self,
        query:      str,
        chunks:     List[Dict],
        lang:       str = None,
        memory:     str | None = None,
    ) -> List[Dict]:
        """
        Build messages untuk Regular RAG (tanpa metadata jurnal kompleks).
        """
        lang = lang if lang is not None else self._detect_language(query)
        tier = self._get_model_tier()

        # Batasi context length
        safe_budget = GROQ_MODEL_SAFE_TOKEN_BUDGET.get(CONFIG["groq_model"], 4_800)
        _usable = safe_budget - FIXED_OVERHEAD_TOKENS
        max_chars = min(int(_usable * 0.30 * 4), CONFIG["context_max_chars"])
        
        if tier == "small":
            max_chars = min(max_chars, 3_000)
        elif tier == "medium":
            max_chars = min(max_chars, 6_000)

        context_parts = []
        used_chars = 0

        for i, chunk in enumerate(chunks, 1):
            part = f"[{i}] {chunk['text']}"
            if used_chars + len(part) > max_chars:
                remaining = max_chars - used_chars
                if remaining > 200:
                    context_parts.append(part[:remaining] + "…")
                break
            context_parts.append(part)
            used_chars += len(part)

        context_str = "\n\n".join(context_parts)
        source_str = "\n".join([f"[{i}] Chunk {i}" for i in range(1, len(chunks) + 1)])

        if lang == "id":
            memory_section = (
                PROMPTS["knowledge_memory_block_id"].format(memory=memory)
                if memory else ""
            )
            prompt_key = {
                "small": "knowledge_system_id_local",
                "medium": "knowledge_system_id_local",
                "large": "knowledge_system_id",
            }[tier]
            system_content = PROMPTS[prompt_key].format(
                memory_section=memory_section,
                context_str=context_str,
                source_str=source_str,
            )
            question_label = "Pertanyaan"
        else:
            memory_section = (
                PROMPTS["knowledge_memory_block_en"].format(memory=memory)
                if memory else ""
            )
            prompt_key = {
                "small": "knowledge_system_en_local",
                "medium": "knowledge_system_en_local",
                "large": "knowledge_system_en",
            }[tier]
            system_content = PROMPTS[prompt_key].format(
                memory_section=memory_section,
                context_str=context_str,
                source_str=source_str,
            )
            question_label = "Question"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{question_label}: {query}"},
        ]

    def _generate_stream(
        self,
        messages:       List[Dict],
        stop_event:     threading.Event = None,
        temperature:    float = None,
        top_p:          float = None,
        max_new_tokens: int   = None,
    ) -> Generator[str, None, None]:
        """
        Generate jawaban — routing otomatis berdasarkan CONFIG["llm_mode"]:
          - "groq"  → Groq API (streaming SSE)
          - "local" → LlamaCpp (streaming token-by-token dari GGUF lokal)

        Parameter messages adalah list OpenAI-style chat messages:
          [{"role": "system"|"user"|"assistant", "content": "..."}]

        stop_event.set() dari luar → hentikan iterasi streaming lebih awal.
        temperature, top_p, max_new_tokens — jika None, pakai CONFIG default.
        """
        _temperature    = temperature    if temperature    is not None else CONFIG["temperature"]
        _top_p          = top_p          if top_p          is not None else CONFIG["top_p"]
        _max_new_tokens = max_new_tokens if max_new_tokens is not None else CONFIG["max_new_tokens"]

        llm_mode = self.models.llm_mode

        if llm_mode == "local":
            yield from self._generate_stream_local(
                messages, stop_event, _temperature, _top_p, _max_new_tokens
            )
        else:
            yield from self._generate_stream_groq(
                messages, stop_event, _temperature, _top_p, _max_new_tokens
            )

    def _generate_stream_groq(
        self,
        messages:       List[Dict],
        stop_event:     threading.Event,
        temperature:    float,
        top_p:          float,
        max_new_tokens: int,
    ) -> Generator[str, None, None]:
        """Generate via Groq API dengan streaming SSE."""
        log.info(
            "[Groq] Generate — model=%s  max_tokens=%d  temperature=%.2f  top_p=%.2f",
            CONFIG["groq_model"], max_new_tokens, temperature, top_p,
        )
        gen_start   = time.perf_counter()
        token_count = 0

        try:
            stream = self.models.groq_client.chat.completions.create(
                model=CONFIG["groq_model"],
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )

            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    log.info("[Groq] Stop event pada token %d", token_count)
                    break

                delta = chunk.choices[0].delta
                text  = getattr(delta, "content", None)
                if text:
                    token_count += 1
                    yield text

        except GeneratorExit:
            log.info("[Groq] GeneratorExit pada token %d", token_count)
        except Exception:
            log.exception("[Groq] Error saat streaming")
            raise
        finally:
            elapsed = time.perf_counter() - gen_start
            log.info("[Groq] ✓ Selesai — %d chunk  %.3fs", token_count, elapsed)

    def _generate_stream_local(
        self,
        messages:       List[Dict],
        stop_event:     threading.Event,
        temperature:    float,
        top_p:          float,
        max_new_tokens: int,
    ) -> Generator[str, None, None]:
        """
        Generate via HuggingFace AutoModelForCausalLM (folder clone dari HF Hub).

        Menggunakan TextIteratorStreamer agar token bisa di-yield satu per satu
        ke SSE tanpa menunggu seluruh respons selesai.

        Alur:
          1. Format messages → apply_chat_template (pakai template bawaan model)
          2. Tokenisasi → tensor GPU
          3. model.generate() dijalankan di thread terpisah (agar tidak blocking)
          4. TextIteratorStreamer di-iterate di thread utama → yield token
          5. stop_event.set() → hentikan iteration lebih awal

        GPU dimonopoli oleh LLM lokal — embedding/reranker/nlp sudah di CPU.
        """
        model_name = os.path.basename(CONFIG.get("local_llm_path", "local"))
        log.info(
            "[LocalLLM] Generate — model=%s  max_tokens=%d  temperature=%.2f  top_p=%.2f",
            model_name, max_new_tokens, temperature, top_p,
        )
        gen_start   = time.perf_counter()
        token_count = 0
        tokenizer   = self.models.local_tokenizer
        model       = self.models.local_llm

        try:
            # ── 1. Format prompt via chat template ───────────────────────────
            # apply_chat_template mengubah list messages ke string prompt
            # yang sesuai dengan format training model (Mistral, Qwen, dll)
            try:
                encoding = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                input_ids = encoding.input_ids.to(model.device)
            except Exception:
                # Fallback: model tidak punya chat template → concat manual
                log.warning("[LocalLLM] Chat template tidak tersedia, gunakan fallback")
                raw = "".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                ) + "ASSISTANT:"
                prompt_ids = tokenizer(raw, return_tensors="pt").input_ids.to(model.device)

            # ── 2. Streamer setup ─────────────────────────────────────────────
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,        # jangan re-yield token prompt
                skip_special_tokens=True,
            )

            # ── 3. Generation config ─────────────────────────────────────────
            gen_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # ── 4. Generate di background thread ─────────────────────────────
            gen_thread = threading.Thread(
                target=model.generate,
                kwargs=gen_kwargs,
                daemon=True,
            )
            gen_thread.start()

            # ── 5. Iterate streamer — yield token ke SSE ──────────────────────
            for text in streamer:
                if stop_event is not None and stop_event.is_set():
                    log.info("[LocalLLM] Stop event pada token %d", token_count)
                    break
                if text:
                    token_count += 1
                    yield text

            gen_thread.join(timeout=5)

        except GeneratorExit:
            log.info("[LocalLLM] GeneratorExit pada token %d", token_count)
        except Exception:
            log.exception("[LocalLLM] Error saat streaming")
            raise
        finally:
            elapsed = time.perf_counter() - gen_start
            log.info("[LocalLLM] ✓ Selesai — %d token  %.3fs", token_count, elapsed)

    # ── Utility ───────────────────────────────────────────────────────────────

    def simple_retrieval(self, query: str, k: int = 5) -> List[Dict]:
        """Testing retrieval tanpa LLM — kembalikan top-k chunk dengan metadata."""
        log.info("simple_retrieval: query=%r  k=%d", query[:80], k)

        emb        = self.models.get_embedding(query)
        candidates = self.chroma.retrieve(emb, k=k)
        enriched   = self.neo4j.enrich(candidates)

        return [
            {
                "sub_judul":    c.sub_judul,
                "konten_chunk": c.konten_chunk[:500] + ("…" if len(c.konten_chunk) > 500 else ""),
                "jurnal":       c.judul_jurnal,
                "penulis":      c.penulis,
                "tahun":        c.tanggal_rilis,
                "halaman":      c.halaman,
                "vector_score": c.vector_score,
            }
            for c in enriched
        ]
    
    def _analyze_image(self, base64_image: str) -> str:
        """Langkah 1: Ekstrak informasi dari gambar menjadi teks deskriptif."""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return "Gambar diterima, tetapi GEMINI_API_KEY tidak ditemukan di environment."
        
        genai.configure(api_key=gemini_api_key)
        # Gunakan model Gemini yang tersedia di akun Anda
        model = genai.GenerativeModel('gemini-2.0-flash') 
        
        try:
            image_data = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(image_data))
        except Exception as e:
            log.error(f"[Vision] Gagal memuat gambar: {e}")
            return "Gambar yang diunggah rusak atau tidak dapat dibaca."
            
        prompt = "Sebagai pakar pertanian, tolong identifikasi dan jelaskan apa yang terlihat pada gambar tanaman ini secara detail, khususnya jika terdapat gejala penyakit, hama, atau kondisi abnormal."
        
        try:
            response = model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            log.error(f"[Gemini] Error analisis gambar: {e}")
            return f"Gagal menganalisis gambar dari server: {e}"


############################################################
# SINGLETON
############################################################

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get-or-create singleton RAGPipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        log.info("Membuat RAGPipeline baru...")
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def reset_pipeline() -> None:
    """
    Paksa destroy dan rebuild seluruh pipeline + model.
    Dipanggil dari app.py ketika terdeteksi instance lama (stale singleton).
    """
    global _rag_pipeline
    log.warning("reset_pipeline() dipanggil — rebuild dari nol.")
    if _rag_pipeline is not None:
        try:
            _rag_pipeline.close()
        except Exception:
            pass
        _rag_pipeline = None
    RAGModels.reset()
    _rag_pipeline = RAGPipeline()


def reload_with_model(mode: str, local_llm_path: str = None) -> None:

    if mode != "groq":
        raise ValueError(
            "Mode 'local' dinonaktifkan. Hanya mode 'groq' yang didukung saat ini."
        )
    log.info(
        "reload_with_model() → mode=groq  model=%s",
        CONFIG.get("groq_model"),
    )
    set_llm_mode("groq", None)

    log.info(
        "reload_with_model() selesai — "
        "embedding=%s  reranker=%s  nlp=%s  llm=groq(%s)",
        CONFIG["embedding_device"],
        CONFIG["reranker_device"],
        "cuda" if CONFIG["nlp_device"] >= 0 else "cpu",
        CONFIG["groq_model"],
    )