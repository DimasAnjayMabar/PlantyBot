import asyncio
import json
import logging
import io
from gtts import gTTS

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db, SessionLocal
from middleware.auth import get_current_session
from models import UserAuth, ChatDetail, Chat
from service.chats import ChatService, KnowledgeService, _get_or_create_event, _cleanup_event, _signal_stop
from validation.chats import (
    CreateTopicSchema,
    RenameTitleSchema,
    SendMessageSchema,
    EditMessageSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chats"])

_SSE_TIMEOUT_SECONDS    = 3600
_SSE_HEARTBEAT_SECONDS  = 15


# =============================================================================
# HELPERS — Serializer
# =============================================================================

def _serialize_detail(d) -> dict:
    return {
        "id":                d.id,
        "chat_id":           d.chat_id,
        "question":          d.question,
        "response":          d.response,
        "processing_status": d.processing_status,
        "created_at":        d.created_at.isoformat(),
        "pipeline_log": {
            "latency_ms":    d.pipeline_log.latency_ms,
            "status":        d.pipeline_log.status,
            "input_tokens":  d.pipeline_log.input_tokens,
            "output_tokens": d.pipeline_log.output_tokens,
            "total_cost":    d.pipeline_log.total_cost,
        } if d.pipeline_log else None,
    }


def _serialize_topic(chat, include_details: bool = False) -> dict:
    data = {
        "id":         chat.id,
        "title":      chat.title,
        "created_at": chat.created_at.isoformat(),
    }
    if include_details:
        data["messages"]       = [_serialize_detail(d) for d in chat.details]
        data["total_messages"] = len(chat.details)
    return data


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_heartbeat() -> str:
    return ": heartbeat\n\n"


# =============================================================================
# TOPICS
# =============================================================================

@router.get("/topics", status_code=status.HTTP_200_OK)
def get_topics(
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        chats = ChatService.get_topics(db, current_session.user_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Daftar topik berhasil diambil.",
                "data": {
                    "topics": [_serialize_topic(c) for c in chats],
                    "total":  len(chats),
                },
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"GET /topics error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengambil topik.")


@router.get("/topics/{chat_id}", status_code=status.HTTP_200_OK)
def get_topic(
    chat_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        chat = ChatService.get_topic(db, current_session.user_id, chat_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Topik berhasil diambil.",
                "data":    _serialize_topic(chat, include_details=True),
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"GET /topics/{chat_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengambil topik.")


@router.delete("/topics/{chat_id}", status_code=status.HTTP_200_OK)
def delete_topic(
    chat_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        ChatService.delete_topic(db, current_session.user_id, chat_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "message": "Topik berhasil dihapus."},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"DELETE /topics/{chat_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menghapus topik.")


@router.patch("/topics/{chat_id}", status_code=status.HTTP_200_OK)
def rename_topic(
    chat_id: int,
    body: RenameTitleSchema,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        chat = ChatService.rename_topic(db, current_session.user_id, chat_id, body.title)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Judul topik berhasil diubah.",
                "data":    _serialize_topic(chat),
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"PATCH /topics/{chat_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengubah judul topik.")


# =============================================================================
# CHAT MESSAGES
# =============================================================================

@router.post("/chat/send", status_code=status.HTTP_202_ACCEPTED)
def send_message(
    body: SendMessageSchema,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    """
    Kirim pertanyaan ke AI.

    Return 202 Accepted dengan detail_id + processing_status='pending'.
    Response (jawaban AI) TIDAK ada di sini — frontend ambil via
    GET /chat/message/{detail_id} setelah SSE memberi sinyal 'done'.
    """
    try:
        detail = ChatService.send_message(
            db,
            current_session.user_id,
            body.chat_id,
            body.question,
            db_factory=SessionLocal,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "success": True,
                "message": "Pertanyaan diterima, sedang diproses.",
                "data": {
                    "id":                detail.id,
                    "chat_id":           detail.chat_id,
                    "question":          detail.question,
                    "processing_status": detail.processing_status,
                    "created_at":        detail.created_at.isoformat(),
                },
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"POST /chat/send error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengirim pesan.")


@router.get("/chat/message/{detail_id}", status_code=status.HTTP_200_OK)
def get_message(
    detail_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    """
    Ambil satu pesan lengkap (beserta jawaban AI) dari DB.

    Frontend memanggil endpoint ini setelah SSE mengirim event 'done'.
    Dengan cara ini, payload SSE hanya berupa sinyal — jawaban AI
    selalu diambil langsung dari DB, bukan dari response JSON.
    """
    try:
        detail = ChatService.get_detail(db, current_session.user_id, detail_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Pesan berhasil diambil.",
                "data":    _serialize_detail(detail),
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"GET /chat/message/{detail_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengambil pesan.")

@router.get("/chat/message/{detail_id}/tts", status_code=status.HTTP_200_OK)
def get_tts_audio(
    detail_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    """
    Mengubah teks jawaban AI (response) menjadi audio (Text-to-Speech).
    Frontend bisa menggunakan endpoint ini langsung di dalam tag <audio src="..."> 
    atau memanggilnya saat tombol TTS ditekan.
    """
    try:
        # 1. Ambil detail pesan dari database
        detail = ChatService.get_detail(db, current_session.user_id, detail_id)
        
        if not detail.response:
            raise HTTPException(status_code=400, detail="Belum ada jawaban dari AI untuk diubah menjadi suara.")

        # 2. Generate Audio menggunakan gTTS (dengan bahasa Indonesia 'id')
        tts = gTTS(text=detail.response, lang='id', slow=False)
        
        # 3. Simpan audio ke dalam buffer memori (agar tidak perlu simpan file fisik di server)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)

        # 4. Stream audio kembali ke frontend
        return StreamingResponse(
            audio_io,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=agribot_response_{detail_id}.mp3"
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"GET /chat/message/{detail_id}/tts error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat men-generate audio TTS.")
    
# Perbaiki bagian akhir stream_response
@router.get("/chat/stream/{detail_id}")
async def stream_response(
    detail_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        detail = ChatService.get_detail(db, current_session.user_id, detail_id)
    except HTTPException as e:
        raise e

    async def event_stream():
        # Cek status awal
        if detail.processing_status in ("done", "failed", "stopped"):   # ← tambah "stopped"
            event_type = (
                "done"    if detail.processing_status == "done"
                else "stopped" if detail.processing_status == "stopped"  # ← baru
                else "error"
            )
            yield _sse_event(event_type, {
                "detail_id":         detail_id,
                "processing_status": detail.processing_status,
            })
            _cleanup_event(detail_id)
            return

        yield _sse_event("waiting", {
            "detail_id": detail_id,
            "processing_status": "pending",
            "message": "Sedang memproses...",
        })

        done_event = _get_or_create_event(detail_id)
        elapsed = 0.0

        try:
            while elapsed < _SSE_TIMEOUT_SECONDS:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(done_event.wait()),
                        timeout=_SSE_HEARTBEAT_SECONDS,
                    )
                    logger.info(f"Event triggered for detail_id={detail_id}")  # ✅ Tambah log
                    break
                except asyncio.TimeoutError:
                    elapsed += _SSE_HEARTBEAT_SECONDS
                    
                    if await request.is_disconnected():
                        logger.info(f"SSE client disconnected — detail_id={detail_id}")
                        _cleanup_event(detail_id)
                        return
                    
                    yield _sse_heartbeat()
            else:
                logger.warning(f"SSE timeout — detail_id={detail_id}")
                yield _sse_event("timeout", {
                    "detail_id": detail_id,
                    "processing_status": "pending",
                    "message": f"Timeout setelah {_SSE_TIMEOUT_SECONDS} detik.",
                })
                _cleanup_event(detail_id)
                return

        except Exception as exc:
            logger.error(f"SSE stream error — detail_id={detail_id}: {exc}")
            _cleanup_event(detail_id)
            return

        if await request.is_disconnected():
            logger.info(f"SSE client disconnected (after done) — detail_id={detail_id}")
            _cleanup_event(detail_id)
            return

        # ✅ Pastikan mengambil status terbaru dari DB
        fresh_db = SessionLocal()
        try:
            fresh_detail = fresh_db.query(ChatDetail).filter_by(id=detail_id).first()
            final_status = fresh_detail.processing_status if fresh_detail else "failed"
            event_type   = (
                "done"    if final_status == "done"
                else "stopped" if final_status == "stopped"   # ← baru
                else "error"
            )
            
            logger.info(f"Sending {event_type} event for detail_id={detail_id}")  # ✅ Log
            yield _sse_event(event_type, {
                "detail_id": detail_id,
                "processing_status": final_status,
            })
        finally:
            fresh_db.close()
            _cleanup_event(detail_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.patch("/chat/edit/{detail_id}", status_code=status.HTTP_202_ACCEPTED)
def edit_message(
    detail_id: int,
    body: EditMessageSchema,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        detail = ChatService.edit_message(
            db,
            current_session.user_id,
            detail_id,
            body.question,
            db_factory=SessionLocal,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "success": True,
                "message": "Pertanyaan diedit, sedang diproses ulang.",
                "data": {
                    "id":                detail.id,
                    "chat_id":           detail.chat_id,
                    "question":          detail.question,
                    "processing_status": detail.processing_status,
                    "created_at":        detail.created_at.isoformat(),
                },
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"PATCH /chat/edit/{detail_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat mengedit pesan.")


@router.post("/chat/regenerate/{detail_id}", status_code=status.HTTP_202_ACCEPTED)
def regenerate_response(
    detail_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        detail = ChatService.regenerate_response(
            db,
            current_session.user_id,
            detail_id,
            db_factory=SessionLocal,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "success": True,
                "message": "Sedang men-generate ulang jawaban.",
                "data": {
                    "id":                detail.id,
                    "chat_id":           detail.chat_id,
                    "question":          detail.question,
                    "processing_status": detail.processing_status,
                    "created_at":        detail.created_at.isoformat(),
                },
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"POST /chat/regenerate/{detail_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat regenerate jawaban.")


@router.delete("/chat/message/{detail_id}", status_code=status.HTTP_200_OK)
def delete_message(
    detail_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        ChatService.delete_message(db, current_session.user_id, detail_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"success": True, "message": "Pesan berhasil dihapus."},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"DELETE /chat/message/{detail_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menghapus pesan.")
    
@router.post("/chat/stop/{detail_id}", status_code=status.HTTP_200_OK)
def stop_generation(
    detail_id: int,
    db: Session = Depends(get_db),
    current_session: UserAuth = Depends(get_current_session),
):
    try:
        detail = ChatService.stop_generation(db, current_session.user_id, detail_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "message": "Pipeline dihentikan.",
                "data": {
                    "id":                detail.id,
                    "chat_id":           detail.chat_id,
                    "processing_status": detail.processing_status,
                },
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"POST /chat/stop/{detail_id} error → {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat menghentikan pipeline.")


# =============================================================================
# KNOWLEDGE BASE — PDF UPLOAD
# =============================================================================

@router.post("/knowledge/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_knowledge_pdf(
    file: UploadFile = File(...),
    judul: str | None = Form(default=None),
    penulis: str | None = Form(default=None),
    tahun: str | None = Form(default=None),
    current_session: UserAuth = Depends(get_current_session),
):
    """
    Upload PDF untuk ditambahkan ke knowledge base chatbot.

    Form fields:
      - file    : file PDF (wajib, maks 50 MB)
      - judul   : judul jurnal/dokumen (opsional)
      - penulis : nama penulis (opsional)
      - tahun   : tahun terbit, e.g. "2024" (opsional)

    Alur:
      1. Validasi tipe file (harus PDF) dan ukuran (maks 50 MB)
      2. Baca bytes file di sini (sebelum UploadFile di-close oleh FastAPI)
      3. Serahkan ke KnowledgeService yang akan:
         a. Simpan file ke ./dataset/
         b. Jalankan embedder di background thread
      4. Return 202 Accepted — embedder berjalan async

    Hanya user yang sudah login yang bisa upload (requires auth).
    """
    # ── Validasi tipe MIME ────────────────────────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Hanya file PDF yang diperbolehkan.",
        )

    # ── Validasi ekstensi sebagai fallback ────────────────────────────────────
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Ekstensi file harus .pdf",
        )

    # ── Baca bytes (maks 50 MB) ───────────────────────────────────────────────
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    file_bytes = await file.read()
    if len(file_bytes) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Ukuran file melebihi batas 50 MB.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File PDF kosong.",
        )

    try:
        result = KnowledgeService.upload_pdf(
            file_bytes  = file_bytes,
            filename    = file.filename or "upload.pdf",
            judul       = judul,
            penulis     = penulis,
            tahun       = tahun,
            user_id     = current_session.user_id,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "success": True,
                "message": "PDF diterima dan sedang diproses ke knowledge base.",
                "data":    result,
            },
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"POST /knowledge/upload error → {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses file PDF.",
        )