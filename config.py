"""
config.py — Konfigurasi Pipeline & Base Prompt LLM
====================================================
File ini berisi:
  1. CONFIG  → semua parameter teknis pipeline (model, DB, retrieval, generation)
  2. PROMPTS → semua base prompt yang digunakan LLM (knowledge, social, memory)

Import di pipeline.py:
  from config import CONFIG, PROMPTS
"""

import os
import torch

# =============================================================================
# KONFIGURASI PIPELINE
# =============================================================================

CONFIG = {
    # ── Model paths ──────────────────────────────────────────────────────────
    "embedding_model":   "intfloat/multilingual-e5-large",
    "reranker_model":    "BAAI/bge-reranker-v2-m3",

    # ── NLP Models (NER / token classification) ───────────────────────────────
    # IndoBERT — digunakan untuk query berbahasa Indonesia
    "nlp_id_model":      "indobenchmark/indobert-base-p1",
    # BERT multilingual — digunakan untuk query berbahasa Inggris
    "nlp_en_model":      "dslim/bert-base-NER",

    # ── LLM Mode ─────────────────────────────────────────────────────────────
    # "groq"  → gunakan Groq API (default) — GPU bebas untuk embedding/reranker/nlp
    # "local" → gunakan model GGUF dari folder ./model/ — GPU hanya untuk LLM lokal
    #
    # Nilai ini diubah secara runtime oleh set_llm_mode() di bawah.
    # JANGAN ubah langsung — gunakan set_llm_mode("groq") atau set_llm_mode("local", path)
    "llm_mode":          "groq",

    # Path model lokal yang aktif (diisi oleh set_llm_mode saat mode "local")
    # Contoh: "./model/Llama-3-8B-Agri-Q4_K_M.gguf"
    "local_llm_path":    None,

    # ── Hardware placement — ditentukan otomatis berdasarkan llm_mode ─────────
    #
    # Mode "groq"  → LLM tidak pakai VRAM → embedding/reranker/nlp bisa di GPU
    # Mode "local" → LLM lokal monopoli GPU → embedding/reranker/nlp dipaksa CPU
    #
    # Nilai ini di-recompute oleh _apply_device_config() setiap kali llm_mode berubah.
    "nlp_device":        0 if torch.cuda.is_available() else -1,
    "embedding_device":  "cuda" if torch.cuda.is_available() else "cpu",
    "reranker_device":   "cuda" if torch.cuda.is_available() else "cpu",

    # ── Groq API ──────────────────────────────────────────────────────────────
    "groq_model":        "llama-3.3-70b-versatile",
    # GROQ_API_KEY dibaca dari environment variable — tidak di-hardcode di sini

    # ── ChromaDB — collection utama + collection memory + collection identity ─
    "chroma_path":       os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"),
    "chroma_collection": "konten_isi",
    "memory_collection": "chat_memory",
    # Collection khusus untuk menyimpan identitas user (nama, preferensi, dll.)
    # Terpisah dari chat_memory agar tidak ikut terhapus saat topic dihapus.
    # Key: 'identity_{user_id}'
    "identity_collection": "user_identity",

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    "neo4j_uri":         "neo4j://127.0.0.1:7687",
    "neo4j_user":        "neo4j",
    "neo4j_password":    "password",

    # ── Pipeline parameters ───────────────────────────────────────────────────
    "chroma_retrieval_k":    20,   # kandidat dari ChromaDB (Tahap 1)
    "context_window":        1,    # window ±N chunk di Neo4j (Tahap 2)
    "reranked_k":            10,   # kandidat setelah reranking (Tahap 3)
    "max_chunks_per_jurnal": 5,    # maks chunk per jurnal di konteks akhir (Tahap 4)
    "final_context_k":       5,    # chunk yang masuk ke prompt LLM (Tahap 4)

    # ── LLM generation — Knowledge pipeline ──────────────────────────────────
    "max_new_tokens":    2048,
    "temperature":       0.2,
    "top_p":             0.95,
    "context_max_chars": 24_000,   # ~6000 token × 4 char/token

    # ── LLM generation — Social pipeline ─────────────────────────────────────
    "social_max_new_tokens": 1024,
    "social_temperature":    0.5,
    "social_top_p":          0.95,

    # ── Memory — Running Summary + Recent Window ──────────────────────────────
    "memory_summary_max_words":  250,
    "memory_summary_model":      "llama-3.3-70b-versatile",
    "memory_summary_max_tokens": 500,
    "memory_recent_window":      5,
}




def _apply_device_config():
    """
    Recompute placement CPU/GPU berdasarkan CONFIG["llm_mode"].

    Mode "groq"  → LLM tidak pakai VRAM lokal → embedding/reranker/nlp bisa GPU
    Mode "local" → LLM lokal monopoli GPU      → embedding/reranker/nlp paksa CPU
    """
    has_cuda = torch.cuda.is_available()
    mode = CONFIG["llm_mode"]

    if mode == "groq":
        CONFIG["embedding_device"] = "cuda" if has_cuda else "cpu"
        CONFIG["reranker_device"]  = "cuda" if has_cuda else "cpu"
        CONFIG["nlp_device"]       = 0 if has_cuda else -1
    elif mode == "local":
        CONFIG["embedding_device"] = "cpu"
        CONFIG["reranker_device"]  = "cpu"
        CONFIG["nlp_device"]       = -1
    else:
        raise ValueError(f"llm_mode tidak dikenal: {mode!r}. Pilih 'groq' atau 'local'.")


def set_llm_mode(mode: str, local_llm_path: str = None):
    """
    Ganti mode LLM secara runtime.

    Args:
        mode           : "groq" atau "local"
        local_llm_path : Path ke file model GGUF (wajib jika mode="local").
                         Contoh: "./model/Llama-3-8B-Q4_K_M.gguf"

    Caller HARUS memanggil reset_pipeline() setelah ini agar singleton di-reload.
    """
    if mode not in ("groq", "local"):
        raise ValueError(f"mode harus 'groq' atau 'local', bukan {mode!r}")
    if mode == "local" and not local_llm_path:
        raise ValueError("local_llm_path wajib diisi saat mode='local'")
    if mode == "local" and local_llm_path and not os.path.isdir(local_llm_path):
        raise ValueError(
            f"local_llm_path harus berupa folder HuggingFace (hasil clone), "
            f"bukan file tunggal: {local_llm_path!r}"
        )

    CONFIG["llm_mode"]       = mode
    CONFIG["local_llm_path"] = local_llm_path if mode == "local" else None
    _apply_device_config()


def list_local_models(model_dir: str = None) -> list:
    """
    Scan folder model/ di root project.
    """
    if model_dir is None:
        # Karena config.py ada di backend/config.py
        # Maka:
        # - backend_dir = /path/to/backend
        # - project_root = /path/to/ (root project)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        
        # Coba beberapa kemungkinan lokasi folder model
        possible_paths = [
            os.path.join(project_root, "model"),           # root/model
            os.path.join(backend_dir, "model"),            # backend/model
            os.path.join(project_root, "..", "model"),     # one level above root
            "./model",                                      # relative current dir
            "../model",                                     # relative one up
        ]
        
        model_dir = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.isdir(abs_path):
                model_dir = abs_path
                print(f"[INFO] Found model directory at: {model_dir}")
                break
        
        if model_dir is None:
            print(f"[ERROR] Model directory not found. Tried: {possible_paths}")
            return []
    
    if not os.path.isdir(model_dir):
        print(f"[ERROR] Model directory does not exist: {model_dir}")
        return []
    
    print(f"[INFO] Scanning models in: {model_dir}")
    print(f"[INFO] Directory contents: {os.listdir(model_dir)}")
    
    models = []
    for name in sorted(os.listdir(model_dir)):
        full_path = os.path.join(model_dir, name)
        
        # Format 1 — HuggingFace folder
        if os.path.isdir(full_path):
            # Cek apakah folder berisi file model
            try:
                files = os.listdir(full_path)
                has_model = any(
                    f.endswith(('.safetensors', '.bin', '.pt', '.pth')) 
                    for f in files
                )
                has_config = (
                    'config.json' in files or 
                    'tokenizer_config.json' in files or
                    'tokenizer.json' in files
                )
                
                if has_config or has_model:
                    models.append({
                        "name": name, 
                        "path": full_path, 
                        "type": "folder"
                    })
                    print(f"[INFO] Added model folder: {name}")
                else:
                    print(f"[WARN] Folder {name} has no model files: {files}")
                    
            except Exception as e:
                print(f"[ERROR] Error reading folder {name}: {e}")
        
        # Format 2 — GGUF single file
        elif name.lower().endswith(".gguf"):
            models.append({
                "name": name, 
                "path": full_path, 
                "type": "gguf"
            })
            print(f"[INFO] Added GGUF file: {name}")
    
    print(f"[INFO] Total models found: {len(models)}")
    return models

# =============================================================================
# BASE PROMPTS LLM
# =============================================================================
# Semua string prompt dikelompokkan di sini.
#
# Prinsip desain:
#   - Base prompt hanya berisi ATURAN RESPONS LLM — tidak memuat data user
#     seperti nama. Informasi identitas user disimpan di ChromaDB collection
#     'user_identity' dan diinjek ke dalam blok memory, bukan ke base prompt.
#
# Placeholder yang diisi via .format() di pipeline.py:
#   {memory_section}  → blok memory yang sudah diformat (termasuk identitas
#                        user jika ada), atau "" jika tidak ada
#   {memory}          → teks memory mentah (dipakai di dalam memory_block)
#   {context_str}     → konteks jurnal gabungan (knowledge pipeline)
#   {source_str}      → daftar sumber referensi (knowledge pipeline)
#   {max_words}       → batas kata summary (memory pipeline)
#   {previous_summary}→ summary lama (memory update prompt)
#   {question}        → pertanyaan user
#   {answer}          → jawaban Agribot

PROMPTS = {

    # -------------------------------------------------------------------------
    # KNOWLEDGE PIPELINE
    # -------------------------------------------------------------------------

    # System prompt utama — Bahasa Indonesia
    "knowledge_system_id": (
        "Anda adalah Agribot, asisten ahli penyakit tanaman yang cerdas. "
        "WAJIB: Jawab dalam Bahasa Indonesia. "
        "Konteks mungkin berbahasa Inggris — terjemahkan istilah teknis jika perlu. "
        "{memory_section}"
        "TUGAS ANDA:\n"
        "1. Selalu gunakan kata 'saya' untuk merujuk diri sendiri.\n"
        "2. Gunakan KONTEKS JURNAL untuk menjawab pertanyaan teknis.\n"
        "3. INGATAN ANDA hanya boleh digunakan sebagai referensi diam-diam — "
        "JANGAN pernah menyebut, mengutip, atau menyinggung isi ingatan di jawaban "
        "kecuali pengguna secara eksplisit bertanya (contoh: 'siapa namaku?', "
        "'apakah kau mengingatku?', 'apa yang pernah aku tanyakan?').\n"
        "4. Jika pengguna bertanya 'siapa namaku?' dan namanya ada di INGATAN ANDA, "
        "jawab langsung dengan namanya.\n"
        "5. JANGAN PERNAH bilang 'saya tidak bisa mengenali individu' atau "
        "'saya tidak memiliki kemampuan mengingat' jika informasinya ada di INGATAN.\n"
        "6. Jika informasi benar-benar tidak ada di jurnal maupun ingatan, barulah nyatakan tidak tahu.\n\n"
        "KONTEKS JURNAL:\n{context_str}\n\n"
        "SUMBER REFERENSI:\n{source_str}"
    ),

    # System prompt utama — English
    "knowledge_system_en": (
        "You are Agribot, an intelligent plant disease expert assistant. "
        "Answer entirely in English. "
        "{memory_section}"
        "YOUR TASKS:\n"
        "1. Always use 'I' to refer to yourself.\n"
        "2. Use the JOURNAL CONTEXT to answer technical questions.\n"
        "3. YOUR MEMORY is for silent reference only — NEVER mention, quote, or allude "
        "to its contents in your answer unless the user explicitly asks "
        "(e.g. 'what's my name?', 'do you remember me?', 'what did I ask before?').\n"
        "4. If the user asks 'what's my name?' and it exists in YOUR MEMORY, "
        "answer directly with their name.\n"
        "5. NEVER say 'I cannot recognize individuals' or 'I don't have the ability "
        "to remember' if the information exists in YOUR MEMORY.\n"
        "6. Only state you don't know if the information is truly absent from both "
        "the journal and memory.\n\n"
        "JOURNAL CONTEXT:\n{context_str}\n\n"
        "REFERENCE SOURCES:\n{source_str}"
    ),

    # Blok memory untuk knowledge pipeline — disisipkan ke {memory_section}
    # Catatan: blok ini memuat identitas user (nama dll.) jika sudah tersimpan
    # di collection 'user_identity', digabung bersama chat_memory sebelum inject.
    "knowledge_memory_block_id": (
        "\n### INGATAN ANDA (IDENTITAS & RIWAYAT PERCAKAPAN) ###\n{memory}\n"
        "Gunakan informasi di atas sebagai ingatan Anda tentang pengguna ini:\n"
        "- Jika ada 'Nama pengguna: X', Anda TAHU nama pengguna — gunakan langsung.\n"
        "- Jika ada RINGKASAN SESI atau PERCAKAPAN TERAKHIR, gunakan untuk konteks berkelanjutan.\n"
        "- Jika hanya ada identitas (nama) tanpa riwayat percakapan, JANGAN sebut "
        "'percakapan sebelumnya' atau 'saya masih ingat obrolan kita' — "
        "cukup kenali pengguna dengan namanya.\n"
        "- JANGAN bilang 'Saya tidak ingat' jika informasinya memang ada di atas.\n"
    ),

    "knowledge_memory_block_en": (
        "\n### YOUR MEMORY (IDENTITY & CONVERSATION HISTORY) ###\n{memory}\n"
        "Use the information above as your memory about this user:\n"
        "- If 'Nama pengguna: X' or 'User name: X' is present, you KNOW the user's name — use it directly.\n"
        "- If there is a SESSION SUMMARY or RECENT CONVERSATION, use it for continuity.\n"
        "- If only identity (name) is present with no conversation history, do NOT mention "
        "'previous conversations' or 'I still remember our chat' — "
        "simply address the user by name.\n"
        "- NEVER say 'I don't remember' if the information is clearly present above.\n"
    ),

    # -------------------------------------------------------------------------
    # SOCIAL PIPELINE
    # -------------------------------------------------------------------------

    # System prompt — Bahasa Indonesia
    "social_system_id": (
        "Kamu adalah Agribot, asisten pertanian yang ramah dan santai. "
        "Selalu gunakan kata 'saya' untuk merujuk dirimu sendiri, JANGAN gunakan 'kami'. "
        "Balas percakapan sosial dengan singkat, hangat, dan natural dalam Bahasa Indonesia. "
        "Jangan sebut tanaman atau pertanian kecuali diminta pengguna. "
        "{memory_section}"
        "ATURAN TENTANG IDENTITAS PENGGUNA:\n"
        "- Jika RIWAYAT di atas memuat 'Nama pengguna: X', kamu SUDAH TAHU nama pengguna — gunakan langsung.\n"
        "- Jika pengguna bertanya 'siapa namaku?' atau 'apakah kau mengenalku?', jawab langsung dengan namanya. "
        "Contoh: 'Ya, namamu adalah Budi. Ada yang bisa saya bantu?'\n"
        "- JANGAN sebut 'percakapan sebelumnya', 'riwayat chat', atau 'saya masih ingat percakapan kita' "
        "jika di RIWAYAT tidak ada isi percakapan — hanya perkenalkan diri dengan namanya saja.\n"
        "- JANGAN PERNAH bilang 'saya tidak bisa mengenali individu', "
        "'namamu adalah user', atau kalimat yang meragukan identitas pengguna.\n\n"
        "Contoh percakapan:\n"
        "Pengguna: halo\n"
        "Agribot: Halo! Bagaimana bisa saya membantu Anda hari ini?\n\n"
        "Pengguna: apakah kau mengenalku?\n"
        "Agribot: Ya, tentu! Namamu adalah Budi. Ada yang bisa saya bantu?\n\n"
        "Pengguna: apa kabar?\n"
        "Agribot: Alhamdulillah baik, terima kasih sudah bertanya! Bagaimana dengan Anda?\n\n"
        "Pengguna: terima kasih\n"
        "Agribot: Sama-sama! Senang bisa membantu. Jangan ragu bertanya lagi ya.\n\n"
        "Pengguna: selamat tinggal\n"
        "Agribot: Selamat tinggal! Semoga hari Anda menyenangkan.\n\n"
        "Pengguna: maaf mengganggu\n"
        "Agribot: Tidak mengganggu sama sekali! Ada yang bisa saya bantu?"
    ),

    # System prompt — English
    "social_system_en": (
        "You are Agribot, a friendly farming assistant. "
        "Reply to casual social messages briefly and naturally in English. "
        "Do not mention plants or farming unless the user asks. "
        "{memory_section}"
        "RULES ABOUT USER IDENTITY:\n"
        "- If the HISTORY above contains 'Nama pengguna: X' or 'User name: X', "
        "you ALREADY KNOW the user's name — use it directly.\n"
        "- If the user asks 'do you know me?' or 'what's my name?', answer directly with their name. "
        "Example: 'Yes, your name is Budi. How can I help you?'\n"
        "- NEVER mention 'previous conversations', 'chat history', or 'I still remember our conversation' "
        "if the HISTORY contains no actual conversation — just greet them by name.\n"
        "- NEVER say 'I cannot recognize individuals', 'your name is user', "
        "or any phrase that doubts the user's identity.\n\n"
        "Example conversations:\n"
        "User: hello\n"
        "Agribot: Hello! How can I help you today?\n\n"
        "User: do you know me?\n"
        "Agribot: Yes, of course! Your name is Budi. How can I help?\n\n"
        "User: how are you?\n"
        "Agribot: I'm doing great, thanks for asking! How about you?\n\n"
        "User: thank you\n"
        "Agribot: You're welcome! Feel free to ask anytime.\n\n"
        "User: goodbye\n"
        "Agribot: Goodbye! Have a wonderful day.\n\n"
        "User: sorry to bother you\n"
        "Agribot: Not a bother at all! What can I help you with?"
    ),

    # Blok memory untuk social pipeline — disisipkan ke {memory_section}
    "social_memory_block_id": "\nRIWAYAT PERCAKAPAN RELEVAN:\n{memory}\n",
    "social_memory_block_en": "\nRELEVANT CONVERSATION HISTORY:\n{memory}\n",

    # -------------------------------------------------------------------------
    # MEMORY PIPELINE — prompt untuk LLM summarizer
    # -------------------------------------------------------------------------

    # Buat summary baru dari percakapan pertama
    "memory_summary_new": (
        "Buat ringkasan percakapan berikut dalam MAKSIMAL {max_words} kata.\n\n"
        "ATURAN PRIORITAS (WAJIB dicantumkan jika ada):\n"
        "1. Topik utama yang dibahas\n"
        "2. Konteks atau pertanyaan user\n\n"
        "Percakapan:\n"
        "Pengguna: {question}\n"
        "Agribot: {answer}\n\n"
        "Ringkasan (maks {max_words} kata, Bahasa Indonesia):"
    ),

    # Update summary yang sudah ada
    "memory_summary_update": (
        "Perbarui ringkasan percakapan berikut dengan menambahkan percakapan baru. "
        "Gunakan MAKSIMAL {max_words} kata.\n\n"
        "ATURAN PRIORITAS (WAJIB dipertahankan, jangan pernah dihapus):\n"
        "1. Topik-topik utama yang sudah dibahas\n"
        "2. Konteks atau pertanyaan terakhir user\n\n"
        "Boleh dikompresi atau dihapus:\n"
        "- Detail teknis yang panjang\n"
        "- Langkah-langkah yang sudah selesai dibahas\n\n"
        "Ringkasan sebelumnya:\n{previous_summary}\n\n"
        "Percakapan baru:\n"
        "Pengguna: {question}\n"
        "Agribot: {answer}\n\n"
        "Ringkasan baru (maks {max_words} kata, Bahasa Indonesia):"
    ),
}