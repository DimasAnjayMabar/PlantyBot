import uvicorn
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from controller.controller_users import router as users_router
from controller.controller_chats import router as chats_router
from pipeline import get_rag_pipeline

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load semua model sebelum server mulai terima request ──
    logger.info("Memuat RAG pipeline saat startup...")
    try:
        get_rag_pipeline()
        logger.info("RAG pipeline siap.")
    except Exception as e:
        logger.critical("Gagal memuat RAG pipeline: %s", e)
        raise  # Hentikan server jika pipeline gagal dimuat

    yield  # Server berjalan di sini

    # ── Shutdown: tutup koneksi Neo4j & bersihkan resource pipeline ──
    logger.info("Server shutdown — menutup RAG pipeline...")
    try:
        from pipeline import _rag_pipeline
        if _rag_pipeline is not None:
            _rag_pipeline.close()
            logger.info("RAG pipeline ditutup.")
        else:
            logger.info("RAG pipeline sudah tidak aktif, skip.")
    except Exception as e:
        logger.warning("Gagal menutup RAG pipeline: %s", e)

app = FastAPI(
    title="Agribot API",
    description="Backend API untuk Chatbot Agribot berbasis AI",
    version="1.0.0",
    lifespan=lifespan,  # ← pasang lifespan di sini
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url} → {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal server error"}
    )

app.include_router(users_router)
app.include_router(chats_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "status": "online",
        "message": "Agribot Backend is running successfully",
        "agent": "FastAPI"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)