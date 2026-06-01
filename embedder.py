#!/usr/bin/env python3
"""
Graph DB: Neo4j | Vector DB: ChromaDB
Structure:
  Neo4j  -> Node Jurnal + Node Isi + Relasi HAS_SECTION & NEXT
  ChromaDB -> collection konten_isi (embedding konten_chunk)

Fitur Baru:
1. Sistem Hashing MD5: Mencegah duplikasi file dengan sidik jari unik
2. Global Chunk Counter: Memperbaiki DuplicateIDError dengan ID unik per dokumen
"""

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import pdfplumber
import chromadb
from neo4j import GraphDatabase

# Import config dari config.py
from config import CONFIG

# RAGModels singleton — embedding model dipinjam dari pipeline yang sudah berjalan,
# tidak ada inisialisasi model baru di sini.
from pipeline import RAGModels

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (ambil dari CONFIG)
# =============================================================================

MAX_TOKENS_PER_CHUNK = CONFIG.get("max_tokens_per_chunk", 512)
CHROMA_PATH = CONFIG["chroma_path"]
CHROMA_COLLECTION = CONFIG["chroma_collection"]
RAW_COLLECTION = CONFIG["raw_collection"]
NEO4J_URI = CONFIG["neo4j_uri"]
NEO4J_USER = CONFIG["neo4j_user"]
NEO4J_PASSWORD = CONFIG["neo4j_password"]
DATASET_PATH = CONFIG.get("dataset_path", "./dataset")
SUBHEADING_SCORE_THRESHOLD = CONFIG.get("subheading_score_threshold", 4)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class JurnalNode:
    """Node Jurnal di Neo4j: {id, judul, doi, penulis, tanggal_rilis, file_hash}"""
    id: str
    judul: str
    doi: Optional[str]
    penulis: str
    tanggal_rilis: str          # format: "YYYY" atau "YYYY-MM-DD"
    source_file: str
    file_hash: str               # MD5 hash untuk deteksi duplikasi

@dataclass
class IsiNode:
    """Node Isi di Neo4j: {id, sub_judul, konten_chunk, halaman}"""
    id: str
    jurnal_id: str
    sub_judul: str
    konten_chunk: str           # teks chunk (juga di-embed ke ChromaDB)
    halaman: int                # halaman awal chunk

# =============================================================================
# FILE HASHING (MD5) UNTUK DETEKSI DUPLIKASI
# =============================================================================

def calculate_file_hash(file_path: str) -> str:
    """
    Menghitung MD5 hash dari file PDF.
    Digunakan untuk mendeteksi apakah file sudah pernah diproses.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Baca file dalam chunk untuk efisiensi memori
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_file_processed(neo4j_driver, file_hash: str) -> bool:
    """
    Mengecek apakah file dengan hash tertentu sudah ada di Neo4j.
    """
    query = """
    MATCH (j:Jurnal {file_hash: $file_hash})
    RETURN count(j) > 0 AS exists
    """
    with neo4j_driver.session() as session:
        result = session.run(query, file_hash=file_hash).single()
        return result["exists"] if result else False

# =============================================================================
# STEP 1 — PDF INGESTION (2-COLUMN AWARE)
# =============================================================================

def _detect_column_split(words: List[Dict], page_width: float) -> Optional[float]:
    """
    Deteksi apakah halaman memiliki 2 kolom dengan mencari gap horizontal
    di area tengah halaman (30%–70% lebar).

    Cara kerja:
    - Bagi lebar halaman menjadi 20 bucket.
    - Bucket yang tidak mengandung kata di area tengah = kandidat gap kolom.
    - Jika ada gap yang konsisten, kembalikan titik tengah gap sebagai pemisah kolom.
    - Jika tidak ditemukan, kembalikan None (halaman 1 kolom).
    """
    if not words or page_width <= 0:
        return None

    bucket_count = 20
    bucket_size = page_width / bucket_count
    buckets = [0] * bucket_count

    for w in words:
        mid_x = (w['x0'] + w['x1']) / 2
        idx = min(int(mid_x / bucket_size), bucket_count - 1)
        buckets[idx] += 1

    # Cari gap (bucket kosong) di area tengah halaman (30%-70%)
    center_start = int(bucket_count * 0.30)
    center_end   = int(bucket_count * 0.70)

    gap_buckets = [
        i for i in range(center_start, center_end)
        if buckets[i] == 0
    ]

    if not gap_buckets:
        return None

    # Ambil tengah gap terpanjang sebagai titik pemisah kolom
    gap_center_idx = gap_buckets[len(gap_buckets) // 2]
    return (gap_center_idx + 0.5) * bucket_size


def _group_words_into_lines(words: List[Dict], page_num: int,
                             page_height: float) -> List[Dict]:
    """
    Kelompokkan words menjadi baris berdasarkan y_position.
    Menyertakan page_height di tiap baris untuk keperluan scoring.
    """
    lines: List[Dict] = []
    current_line: List[Dict] = []
    current_y: Optional[float] = None

    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        if current_y is None:
            current_y = word['top']
            current_line = [word]
        elif abs(word['top'] - current_y) < 5:
            current_line.append(word)
        else:
            if current_line:
                lines.append(_build_line_dict(current_line, page_num, current_y, page_height))
            current_y = word['top']
            current_line = [word]

    if current_line:
        lines.append(_build_line_dict(current_line, page_num, current_y or 0, page_height))

    return lines


def parse_pdf_to_lines(pdf_path: str) -> List[Dict]:
    """
    Extract lines dari PDF dengan dukungan layout 2 kolom.

    Strategi 2-kolom:
    - Untuk tiap halaman, deteksi apakah ada pemisah kolom (gap horizontal).
    - Jika ya: pisahkan words menjadi kolom kiri dan kanan, proses masing-masing
      secara terpisah (kiri dulu, baru kanan) agar urutan kalimat tetap benar.
    - Jika tidak: proses seperti biasa (sort by top, x0).
    """
    all_lines: List[Dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            if not words:
                continue

            page_width  = float(page.width)
            page_height = float(page.height)

            # Deteksi titik pemisah kolom
            col_split = _detect_column_split(words, page_width)

            if col_split is not None:
                # Layout 2 kolom: proses kolom kiri → kanan secara terpisah
                left_words  = [w for w in words if w['x1'] <= col_split]
                right_words = [w for w in words if w['x0'] >  col_split]

                all_lines.extend(_group_words_into_lines(left_words,  page_num, page_height))
                all_lines.extend(_group_words_into_lines(right_words, page_num, page_height))
            else:
                # Layout 1 kolom: proses normal
                all_lines.extend(_group_words_into_lines(words, page_num, page_height))

    return all_lines


def _build_line_dict(words: List[Dict], page_num: int,
                     y_pos: float, page_height: float = 0.0) -> Dict:
    line_text = ' '.join(w['text'] for w in words)
    avg_size  = sum(float(w.get('height', 10)) for w in words) / len(words)
    is_bold   = any('bold' in str(w.get('fontname', '')).lower() for w in words)
    return {
        'text':        line_text.strip(),
        'page':        page_num,
        'font_size':   avg_size,
        'is_bold':     is_bold,
        'y_position':  y_pos,
        'page_height': page_height,   # diperlukan untuk scoring subheading
    }

# =============================================================================
# STEP 2 — REMOVE BOILERPLATE
# =============================================================================

BOILERPLATE_PATTERNS = [
    r"Prosiding SEMNAS BIO",
    r"ISSN",
    r"Quo Vadis",
    r"^\d+$",
    r"^halaman\s+\d+",
    r"©\s*\d{4}",
    r"www\.",
    r"http://",
    r"https://",
]

def is_boilerplate(text: str) -> bool:
    text = text.strip()
    if len(text) < 3:
        return True
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_lines(lines: List[Dict]) -> List[Dict]:
    return [l for l in lines if not is_boilerplate(l["text"])]

# =============================================================================
# STEP 3 — HEADING DETECTION (HEURISTIC SCORING)
# =============================================================================

# Threshold skor minimum untuk dianggap sub-judul
SUBHEADING_SCORE_THRESHOLD = CONFIG.get("subheading_score_threshold", 4)

def is_all_caps(text: str) -> bool:
    """Cek apakah teks dominan UPPERCASE (>80% huruf kapital)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


def compute_dominant_font_size(lines: List[Dict]) -> float:
    """
    Hitung modus (font size paling sering muncul) dari seluruh baris.
    Digunakan sebagai ukuran font bodi normal.
    """
    from statistics import mode
    sizes = [round(l["font_size"], 1) for l in lines if l.get("font_size")]
    if not sizes:
        return 10.0
    try:
        return mode(sizes)
    except Exception:
        return sorted(sizes)[len(sizes) // 2]


def compute_normal_line_gap(lines: List[Dict]) -> float:
    """
    Hitung jarak baris normal (median selisih y_position antar baris berurutan
    pada halaman yang sama). Digunakan sebagai acuan jarak vertikal normal.
    """
    gaps = []
    for i in range(1, len(lines)):
        if lines[i]["page"] == lines[i - 1]["page"]:
            gap = lines[i]["y_position"] - lines[i - 1]["y_position"]
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 12.0
    gaps_sorted = sorted(gaps)
    return gaps_sorted[len(gaps_sorted) // 2]


def score_subheading(line: Dict,
                     prev_line: Optional[Dict],
                     dominant_font_size: float,
                     normal_line_gap: float) -> int:
    """
    Hitung skor heuristik untuk menentukan apakah baris adalah sub-judul.

    Aturan pembobotan:
      +2  : Teks Bold
      +2  : Jarak vertikal (top) > 1.5x jarak baris normal (baris berdiri sendiri)
      +2  : Ukuran font > ukuran bodi dominan
      +1  : Tidak diakhiri tanda titik (.)
      +1  : Format UPPERCASE
      -2  : Panjang karakter > 60 (asumsi paragraf)

    Threshold: skor >= SUBHEADING_SCORE_THRESHOLD -> sub-judul baru
    """
    text = line["text"].strip()
    score = 0

    # +2: Bold
    if line.get("is_bold"):
        score += 2

    # +2: Jarak vertikal > 1.5x normal (isolated line)
    if prev_line and prev_line["page"] == line["page"]:
        gap = line["y_position"] - prev_line["y_position"]
        if gap > 1.5 * normal_line_gap:
            score += 2
    else:
        # Baris pertama di halaman atau halaman berbeda -> dianggap isolated
        score += 2

    # +2: Font size lebih besar dari ukuran bodi dominan
    if line.get("font_size", 0) > dominant_font_size:
        score += 2

    # +1: Tidak diakhiri titik
    if not text.endswith('.'):
        score += 1

    # +1: UPPERCASE
    if is_all_caps(text):
        score += 1

    # -2: Panjang karakter > 60 (kemungkinan paragraf bukan judul)
    if len(text) > 60:
        score -= 2

    # -1: Subheading di bagian bawah halaman (y > 90% tinggi halaman).
    # Subheading yang "tergantung" di akhir halaman biasanya bukan subheading asli.
    page_height = line.get("page_height", 0)
    if page_height > 0 and line["y_position"] > (page_height * 0.9):
        score -= 1

    return score


def is_subheading(line: Dict,
                  prev_line: Optional[Dict],
                  dominant_font_size: float,
                  normal_line_gap: float) -> bool:
    """
    Return True jika baris terdeteksi sebagai sub-judul berdasarkan skor heuristik.
    Juga skip baris yang terlalu pendek (< 3 karakter).
    """
    text = line["text"].strip()
    if len(text) < 3:
        return False

    score = score_subheading(line, prev_line, dominant_font_size, normal_line_gap)
    return score >= SUBHEADING_SCORE_THRESHOLD

# =============================================================================
# STEP 4 — TOKEN COUNTING & CHUNKING
# =============================================================================

def count_tokens(text: str) -> int:
    return len(text) // 4

def split_text_word_safe(text: str, max_tokens: int) -> List[str]:
    chunks: List[str] = []
    current_chunk = ""
    sentences = re.split(r'([.!?]+\s+)', text)

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = sentence + delimiter

        if current_chunk and count_tokens(current_chunk + full_sentence) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = full_sentence
        else:
            current_chunk += full_sentence

        if count_tokens(current_chunk) > max_tokens:
            words = current_chunk.split()
            temp_chunk = ""
            for word in words:
                if count_tokens(temp_chunk + " " + word) > max_tokens:
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        chunks.append(word)
                        temp_chunk = ""
                else:
                    temp_chunk += " " + word if temp_chunk else word
            current_chunk = temp_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# =============================================================================
# STEP 5 — BUILD ISI NODES (DENGAN GLOBAL CHUNK COUNTER)
# =============================================================================

def deterministic_id(jurnal_id: str, sub_judul: str, chunk_index: int) -> str:
    """
    Membuat ID deterministik untuk IsiNode.
    Dengan chunk_index global per dokumen, ID akan selalu unik.
    """
    base = f"{jurnal_id}:{sub_judul}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def build_isi_nodes(lines: List[Dict], jurnal_id: str) -> List[IsiNode]:
    """
    Build list of IsiNode from PDF lines menggunakan heuristic scoring.

    Langkah:
    1. Pre-compute dominant font size & normal line gap dari seluruh baris.
    2. Iterasi baris; tiap baris dievaluasi dengan score_subheading().
    3. Jika skor >= SUBHEADING_SCORE_THRESHOLD -> flush buffer, mulai section baru.
    4. Tiap section di-chunk dengan split_text_word_safe() -> IsiNode.
    
    Perbaikan: Menggunakan global_chunk_counter untuk memastikan semua ID unik.
    """
    if not lines:
        return []

    # Pre-compute statistik layout untuk scoring
    dominant_font_size = compute_dominant_font_size(lines)
    normal_line_gap = compute_normal_line_gap(lines)

    print(f"  [Heuristic] dominant_font_size={dominant_font_size:.1f}  "
          f"normal_line_gap={normal_line_gap:.1f}  "
          f"threshold={SUBHEADING_SCORE_THRESHOLD}")

    isi_nodes: List[IsiNode] = []
    current_heading = "Pendahuluan"
    current_buffer: List[Dict] = []
    current_page_start = lines[0]["page"]
    
    # Global chunk counter untuk memastikan ID unik di seluruh dokumen
    global_chunk_counter = 0

    def flush_buffer():
        nonlocal current_heading, global_chunk_counter
        if not current_buffer:
            return
        full_text = " ".join(l["text"] for l in current_buffer)
        chunks = split_text_word_safe(full_text, MAX_TOKENS_PER_CHUNK)
        for chunk in chunks:
            node_id = deterministic_id(jurnal_id, current_heading, global_chunk_counter)
            isi_nodes.append(IsiNode(
                id=node_id,
                jurnal_id=jurnal_id,
                sub_judul=current_heading,
                konten_chunk=chunk,
                halaman=current_page_start,
            ))
            global_chunk_counter += 1

    for i, line in enumerate(lines):
        prev_line = lines[i - 1] if i > 0 else None

        if is_subheading(line, prev_line, dominant_font_size, normal_line_gap):
            flush_buffer()
            current_heading = line["text"].strip().rstrip(':.').strip()
            current_buffer = []
            current_page_start = line["page"]
        else:
            current_buffer.append(line)

    flush_buffer()
    return isi_nodes

# =============================================================================
# EMBEDDING — dipinjam dari RAGModels singleton
# =============================================================================

# =============================================================================
# NEO4J INGESTOR
# =============================================================================

class Neo4jIngestor:
    """
    Manages Neo4j graph ingestion.
    Nodes  : Jurnal {id, judul, doi, penulis, tanggal_rilis, file_hash}
             Isi    {id, sub_judul, konten_chunk, halaman}
    Edges  : (Jurnal)-[:HAS_SECTION]->(Isi)
             (Isi)-[:NEXT]->(Isi)
    """

    def __init__(self,
                 uri: str = NEO4J_URI,
                 user: str = NEO4J_USER,
                 password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"Neo4j connected: {uri}")

    def close(self):
        self.driver.close()

    # ── Constraints (idempotent) ──────────────────────────────

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:Jurnal) REQUIRE j.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Isi) REQUIRE i.id IS UNIQUE")
        print("Neo4j constraints ensured.")

    # ── Jurnal node ───────────────────────────────────────────

    def ingest_jurnal(self, jurnal: JurnalNode):
        query = """
        MERGE (j:Jurnal {id: $id})
        SET j.judul        = $judul,
            j.doi          = $doi,
            j.penulis      = $penulis,
            j.tanggal_rilis = $tanggal_rilis,
            j.file_hash    = $file_hash,
            j.source_file  = $source_file
        """
        with self.driver.session() as session:
            session.run(query,
                        id=jurnal.id,
                        judul=jurnal.judul,
                        doi=jurnal.doi or "",
                        penulis=jurnal.penulis,
                        tanggal_rilis=jurnal.tanggal_rilis,
                        file_hash=jurnal.file_hash,
                        source_file=jurnal.source_file)
        print(f"  ✓ Neo4j: Jurnal ingested ({jurnal.id})")

    # ── Isi nodes + HAS_SECTION + NEXT ───────────────────────

    def ingest_isi_nodes(self, isi_nodes: List[IsiNode]):
        if not isi_nodes:
            return

        # Batch upsert Isi nodes
        batch = [
            {
                "id": n.id,
                "jurnal_id": n.jurnal_id,
                "sub_judul": n.sub_judul,
                "konten_chunk": n.konten_chunk,
                "halaman": n.halaman,
            }
            for n in isi_nodes
        ]

        with self.driver.session() as session:
            # Create/update Isi nodes
            session.run("""
                UNWIND $batch AS row
                MERGE (i:Isi {id: row.id})
                SET i.sub_judul    = row.sub_judul,
                    i.konten_chunk = row.konten_chunk,
                    i.halaman      = row.halaman
            """, batch=batch)

            # Create (Jurnal)-[:HAS_SECTION]->(Isi)
            session.run("""
                UNWIND $batch AS row
                MATCH (j:Jurnal {id: row.jurnal_id})
                MATCH (i:Isi    {id: row.id})
                MERGE (j)-[:HAS_SECTION]->(i)
            """, batch=batch)

            # Create (Isi)-[:NEXT]->(Isi) — batch dengan UNWIND
            next_pairs = [
                {"from_id": isi_nodes[i].id, "to_id": isi_nodes[i + 1].id}
                for i in range(len(isi_nodes) - 1)
            ]
            if next_pairs:
                session.run("""
                    UNWIND $pairs AS pair
                    MATCH (a:Isi {id: pair.from_id})
                    MATCH (b:Isi {id: pair.to_id})
                    MERGE (a)-[:NEXT]->(b)
                """, pairs=next_pairs)

        print(f"  ✓ Neo4j: {len(isi_nodes)} Isi nodes ingested with HAS_SECTION & NEXT edges")

# =============================================================================
# CHROMADB INGESTOR
# =============================================================================

class ChromaIngestor:
    """
    Manages ChromaDB vector ingestion.
    Collection: konten_isi
    Embedding : konten_chunk
    Metadata  : {isi_id, jurnal_id}
    """

    def __init__(self, persist_directory: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"description": "Embeddings konten_chunk dari Node Isi"}
        )
        print(f"ChromaDB initialized at: {persist_directory}")

    def ingest_isi_nodes(self, isi_nodes: List[IsiNode], rag_models: RAGModels,
                         judul_jurnal: str = ""):
        """
        Ingest IsiNode ke ChromaDB dengan contextualized embedding.

        Teks yang di-embed = "{judul_jurnal} | {sub_judul} | {konten_chunk}"
        Embedding dihasilkan via rag_models.embed_batch_safe() — thread-safe,
        berbagi GPU dengan RAG pipeline tanpa duplikasi instance model.

        ChromaDB `documents` tetap menyimpan konten_chunk murni (untuk retrieval),
        sedangkan embedding dihasilkan dari teks yang sudah diperkaya konteks.
        """
        if not isi_nodes:
            return

        ids       = [n.id for n in isi_nodes]
        documents = [n.konten_chunk for n in isi_nodes]

        # Teks yang di-embed: diperkaya dengan konteks judul jurnal + sub_judul
        texts_to_embed = [
            f"{judul_jurnal} | {n.sub_judul} | {n.konten_chunk}"
            for n in isi_nodes
        ]
        embeddings = rag_models.embed_batch_safe(texts_to_embed)

        metadatas = [
            {"isi_id": n.id, "jurnal_id": n.jurnal_id}
            for n in isi_nodes
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "ChromaDB: %d konten_chunk ingested ke '%s' (contextualized embedding)",
            len(isi_nodes), CHROMA_COLLECTION,
        )

# =============================================================================
# CHROMADB INGESTOR — RAW (tanpa heading detection & Neo4j)
# =============================================================================

def is_file_processed_raw(chroma_client, file_hash: str) -> bool:
    """
    Cek apakah file dengan hash tertentu sudah ada di collection konten_isi_raw.
    Dipakai sebagai pengganti cek Neo4j untuk pipeline raw.
    """
    try:
        col = chroma_client.get_or_create_collection(RAW_COLLECTION)
        results = col.get(where={"file_hash": file_hash}, limit=1, include=[])
        return len(results["ids"]) > 0
    except Exception:
        return False


class ChromaIngestorRaw:
    """
    Manages ChromaDB vector ingestion untuk mode RAW.

    Pipeline raw TIDAK menggunakan Neo4j dan TIDAK melakukan heading detection.
    Alur: PDF → 2-column detection → boilerplate removal → flat chunking
          → embed teks mentah → konten_isi_raw

    Collection : konten_isi_raw  (terpisah dari konten_isi improved)
    Embedding  : teks chunk murni — TANPA prefix judul/sub_judul
    Metadata   : {file_hash, jurnal_id, chunk_index, source_file}
    Duplikasi  : dicek via metadata file_hash di ChromaDB (tanpa Neo4j)
    """

    def __init__(self, persist_directory: str = CHROMA_PATH, chroma_client=None):
        # Jika client sudah ada (dipinjam dari pipeline singleton), pakai langsung.
        # Jika tidak (misal dijalankan standalone via CLI), buat baru dari persist_directory.
        if chroma_client is not None:
            self.client = chroma_client
            print(f"ChromaDB (raw) memakai shared client  collection='{RAW_COLLECTION}'")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"ChromaDB (raw) initialized at: {persist_directory}  "
                  f"collection='{RAW_COLLECTION}'")
        self.collection = self.client.get_or_create_collection(
            name=RAW_COLLECTION,
            metadata={"description": "Embeddings raw — flat chunks tanpa heading/Neo4j"}
        )

    def ingest_chunks(self,
                      chunks: List[str],
                      file_hash: str,
                      jurnal_id: str,
                      source_file: str,
                      rag_models: RAGModels):
        """
        Ingest flat list of text chunks ke ChromaDB collection raw.

        ID  : deterministik dari jurnal_id + index → tidak ada duplikat per file
        Embed: teks chunk murni (tanpa prefix konteks)
        """
        if not chunks:
            return

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{jurnal_id}:raw:{i}"))
            for i in range(len(chunks))
        ]
        embeddings = rag_models.embed_batch_safe(chunks)
        metadatas  = [
            {
                "file_hash":   file_hash,
                "jurnal_id":   jurnal_id,
                "chunk_index": i,
                "source_file": source_file,
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "ChromaDB (raw): %d chunks ingested ke '%s'",
            len(chunks), RAW_COLLECTION,
        )

# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(pdf_path: str,
                 jurnal_metadata: Dict,
                 rag_models: RAGModels,
                 neo4j: Neo4jIngestor,
                 chroma: ChromaIngestor) -> Optional[Dict]:
    """
    Main pipeline: PDF -> IsiNode -> Neo4j (graph) + ChromaDB (vector)

    jurnal_metadata keys: judul, doi, penulis, tanggal_rilis

    rag_models  : RAGModels singleton dari pipeline — embedding model dipinjam,
                  tidak ada inisialisasi baru.
    neo4j       : Neo4jIngestor — koneksi ke graph DB.
    chroma      : ChromaIngestor — koneksi ke vector DB.

    Returns:
        Dict hasil jika file baru diproses
        None jika file sudah ada (duplikat berdasarkan MD5 hash)
    """
    logger.info("Processing: %s", pdf_path)

    # Hitung hash file untuk deteksi duplikasi
    file_hash = calculate_file_hash(pdf_path)
    logger.info("File hash (MD5): %s", file_hash)

    # Cek apakah file sudah pernah diproses
    if is_file_processed(neo4j.driver, file_hash):
        logger.warning(
            "File sudah pernah diproses (hash: %s...). Melewati...", file_hash[:8]
        )
        return None

    # Step 1: Parse PDF
    logger.info("Step 1: Parsing PDF...")
    lines = parse_pdf_to_lines(pdf_path)
    logger.info("  Extracted %d lines", len(lines))

    # Step 2: Clean boilerplate
    logger.info("Step 2: Removing boilerplate...")
    lines = clean_lines(lines)
    logger.info("  Retained %d lines after cleaning", len(lines))

    # Step 3: Create Jurnal node
    jurnal_id = str(uuid.uuid4())
    jurnal = JurnalNode(
        id=jurnal_id,
        judul=jurnal_metadata.get("judul", "Unknown"),
        doi=jurnal_metadata.get("doi"),
        penulis=jurnal_metadata.get("penulis", "Unknown"),
        tanggal_rilis=str(jurnal_metadata.get("tanggal_rilis", "2024")),
        source_file=pdf_path,
        file_hash=file_hash,
    )

    # Step 4: Build Isi nodes
    logger.info("Step 3: Building Isi nodes...")
    isi_nodes = build_isi_nodes(lines, jurnal.id)
    logger.info("  Created %d Isi nodes", len(isi_nodes))

    # Step 5: Ingest to Neo4j
    logger.info("Step 4: Ingesting to Neo4j...")
    neo4j.ingest_jurnal(jurnal)
    neo4j.ingest_isi_nodes(isi_nodes)

    # Step 6: Ingest to ChromaDB (pakai shared embedding model via lock)
    logger.info("Step 5: Ingesting to ChromaDB (shared embedding model)...")
    chroma.ingest_isi_nodes(isi_nodes, rag_models, judul_jurnal=jurnal.judul)

    headings: Dict[str, int] = {}
    for node in isi_nodes:
        headings[node.sub_judul] = headings.get(node.sub_judul, 0) + 1

    return {
        "jurnal": jurnal,
        "isi_nodes": isi_nodes,
        "stats": {
            "total_isi_nodes": len(isi_nodes),
            "unique_headings": len(headings),
            "avg_chunks_per_heading": len(isi_nodes) / len(headings) if headings else 0,
        },
    }


def run_pipeline_with_shared_resources(
    pdf_path: str,
    jurnal_metadata: Dict,
) -> Optional[Dict]:
    """
    Entry point untuk service backend saat upload PDF via API (mode IMPROVED).

    Meminjam semua resource dari singleton yang sudah berjalan:
      - RAGModels  → embedding model (shared, thread-safe via lock)
      - Neo4j      → driver dari RAGPipeline
      - ChromaDB   → client dari RAGPipeline

    Tidak membuat koneksi atau model baru — zero VRAM overhead.

    Dipanggil dari background thread di service backend (bukan dari main()),
    sehingga aman untuk berjalan bersamaan dengan RAG query.
    """
    from pipeline import get_rag_pipeline

    pipeline   = get_rag_pipeline()
    rag_models = pipeline.models  # RAGModels singleton

    neo4j = Neo4jIngestor(
        uri      = NEO4J_URI,
        user     = NEO4J_USER,
        password = NEO4J_PASSWORD,
    )
    chroma = ChromaIngestor(persist_directory=CHROMA_PATH)

    try:
        return run_pipeline(pdf_path, jurnal_metadata, rag_models, neo4j, chroma)
    finally:
        neo4j.close()


def run_pipeline_raw(pdf_path: str,
                     jurnal_metadata: Dict,
                     rag_models: RAGModels,
                     chroma_raw: ChromaIngestorRaw) -> Optional[Dict]:
    """
    Pipeline RAW: PDF → 2-column detection → boilerplate removal
                  → flat chunking → embed → konten_isi_raw

    TIDAK menggunakan Neo4j dan TIDAK melakukan heading detection.
    Duplikasi dicek via metadata file_hash di ChromaDB.

    Returns:
        Dict hasil jika file baru diproses
        None jika file sudah ada (duplikat berdasarkan MD5 hash)
    """
    logger.info("[RAW] Processing: %s", pdf_path)

    # ── Cek duplikasi via ChromaDB (tanpa Neo4j) ──────────────────────────
    file_hash = calculate_file_hash(pdf_path)
    logger.info("[RAW] File hash (MD5): %s", file_hash)

    if is_file_processed_raw(chroma_raw.client, file_hash):
        logger.warning(
            "[RAW] File sudah pernah diproses (hash: %s...). Melewati...", file_hash[:8]
        )
        return None

    # ── Step 1: Parse PDF dengan 2-column detection ───────────────────────
    logger.info("[RAW] Step 1: Parsing PDF (2-column aware)...")
    lines = parse_pdf_to_lines(pdf_path)
    logger.info("[RAW]   Extracted %d lines", len(lines))

    # ── Step 2: Hapus boilerplate ─────────────────────────────────────────
    logger.info("[RAW] Step 2: Removing boilerplate...")
    lines = clean_lines(lines)
    logger.info("[RAW]   Retained %d lines after cleaning", len(lines))

    if not lines:
        logger.warning("[RAW] Tidak ada konten setelah cleaning — pipeline berhenti.")
        return None

    # ── Step 3: Flat chunking — gabung semua teks, lalu potong per N token ─
    # Tidak ada heading detection; seluruh teks diperlakukan sebagai satu alur.
    logger.info("[RAW] Step 3: Flat chunking...")
    full_text = " ".join(l["text"] for l in lines)
    chunks    = split_text_word_safe(full_text, MAX_TOKENS_PER_CHUNK)
    logger.info("[RAW]   Created %d flat chunks", len(chunks))

    # ── Step 4: Embed & ingest ke ChromaDB raw ────────────────────────────
    jurnal_id = str(uuid.uuid4())
    logger.info("[RAW] Step 4: Ingesting to ChromaDB RAW...")
    chroma_raw.ingest_chunks(
        chunks      = chunks,
        file_hash   = file_hash,
        jurnal_id   = jurnal_id,
        source_file = pdf_path,
        rag_models  = rag_models,
    )

    return {
        "jurnal_id":   jurnal_id,
        "source_file": pdf_path,
        "stats": {
            "total_lines":  len(lines),
            "total_chunks": len(chunks),
        },
    }


def run_pipeline_with_shared_resources_raw(
    pdf_path: str,
    jurnal_metadata: Dict,
) -> Optional[Dict]:
    """
    Entry point untuk service backend saat upload PDF via API (mode RAW).

    Tidak membuat koneksi Neo4j — hanya ChromaDB.
    Collection: konten_isi_raw
    """
    from pipeline import get_rag_pipeline

    pipeline   = get_rag_pipeline()
    rag_models = pipeline.models  # RAGModels singleton

    # Pinjam client ChromaDB dari singleton pipeline — jangan buat instance baru
    # agar tidak konflik "An instance of Chroma already exists with different settings"
    chroma_raw = ChromaIngestorRaw(chroma_client=pipeline.chroma.client)
    return run_pipeline_raw(pdf_path, jurnal_metadata, rag_models, chroma_raw)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global MAX_TOKENS_PER_CHUNK  # harus di baris pertama sebelum assignment apapun
    import argparse

    parser = argparse.ArgumentParser(
        description="Disease Journal PDF Embedder — Neo4j + ChromaDB"
    )
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--chroma", default=CHROMA_PATH)
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    parser.add_argument("--file", help="Process single PDF instead of entire dataset")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_CHUNK)
    parser.add_argument("--force", action="store_true",
                       help="Force reprocess even if file already exists (ignore hash check)")
    args = parser.parse_args()

    # Override MAX_TOKENS_PER_CHUNK jika di CLI
    MAX_TOKENS_PER_CHUNK = args.max_tokens

    # Saat dijalankan via CLI (standalone), RAGModels diinisialisasi di sini.
    # Saat dipanggil oleh service backend, gunakan run_pipeline_with_shared_resources()
    # agar tidak ada duplikasi model di VRAM.
    print("Initializing RAGModels (embedding + reranker)...")
    rag_models = RAGModels()

    print("Initializing Neo4j...")
    neo4j = Neo4jIngestor(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    neo4j.create_constraints()

    print("Initializing ChromaDB...")
    chroma = ChromaIngestor(persist_directory=args.chroma)

    # Find PDFs
    if args.file:
        pdf_files = [Path(args.file)]
    else:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset path not found: {dataset_path}")
            neo4j.close()
            return
        pdf_files = list(dataset_path.glob("**/*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        neo4j.close()
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) to process")

    all_results = []
    skipped_files = 0

    for pdf_file in pdf_files:
        try:
            jurnal_metadata = {
                "judul":         pdf_file.stem,
                "doi":           None,
                "penulis":       "Unknown Author",
                "tanggal_rilis": "2024",
            }

            if args.force:
                file_hash = calculate_file_hash(str(pdf_file))
                logger.info("Force processing: %s  hash=%s", pdf_file, file_hash)
                lines = parse_pdf_to_lines(str(pdf_file))
                lines = clean_lines(lines)
                jurnal_id = str(uuid.uuid4())
                jurnal = JurnalNode(
                    id=jurnal_id,
                    judul=jurnal_metadata["judul"],
                    doi=jurnal_metadata.get("doi"),
                    penulis=jurnal_metadata["penulis"],
                    tanggal_rilis=jurnal_metadata["tanggal_rilis"],
                    source_file=str(pdf_file),
                    file_hash=file_hash,
                )
                isi_nodes = build_isi_nodes(lines, jurnal.id)
                neo4j.ingest_jurnal(jurnal)
                neo4j.ingest_isi_nodes(isi_nodes)
                chroma.ingest_isi_nodes(isi_nodes, rag_models, judul_jurnal=jurnal.judul)
                all_results.append({
                    "jurnal": jurnal,
                    "isi_nodes": isi_nodes,
                    "stats": {"total_isi_nodes": len(isi_nodes)},
                })
            else:
                result = run_pipeline(
                    str(pdf_file), jurnal_metadata, rag_models, neo4j, chroma
                )
                if result:
                    all_results.append(result)
                else:
                    skipped_files += 1

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()

    neo4j.close()

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Processed       : {len(all_results)} file(s)")
    if skipped_files > 0:
        print(f"Skipped         : {skipped_files} file(s) (already exist)")
    total_nodes = sum(r["stats"]["total_isi_nodes"] for r in all_results)
    print(f"Total Isi nodes : {total_nodes}")
    print(f"\nNeo4j   : {args.neo4j_uri}")
    print(f"ChromaDB: {args.chroma}  |  collection: {CHROMA_COLLECTION}")
    print("\nGraph structure:")
    print("  (Jurnal)-[:HAS_SECTION]->(Isi)")
    print("  (Isi)-[:NEXT]->(Isi)")

if __name__ == "__main__":
    main()