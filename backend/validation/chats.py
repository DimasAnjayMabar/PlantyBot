from pydantic import BaseModel, field_validator


class RenameTitleSchema(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Judul tidak boleh kosong.")
        return v.strip()


class SendMessageSchema(BaseModel):
    chat_id: int | None = None  # null = auto-create topic baru
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        return v.strip()


class EditMessageSchema(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Pertanyaan tidak boleh kosong.")
        return v.strip()