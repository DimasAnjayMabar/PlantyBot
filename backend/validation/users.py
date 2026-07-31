from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from typing import List

# --- 1. REGISTRASI ---
class RegisterSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="petanikeren")
    email: EmailStr = Field(..., max_length=100, example="user@agribot.com")
    password: str = Field(..., min_length=8, example="password123")
    name: str = Field(..., min_length=2, max_length=100, example="Budi Santoso")

# --- 2. LOGIN ---
class LoginSchema(BaseModel):
    # Bisa diisi username atau email
    identifier: str = Field(..., example="petanikeren atau user@agribot.com")
    password: str = Field(..., example="password123")

# --- 3. REQUEST OTP ---
class RequestOtpSchema(BaseModel):
    email: EmailStr = Field(..., example="user@agribot.com")

# --- 4. VERIFIKASI OTP ---
class VerifyOtp(BaseModel):
    email: EmailStr = Field(..., example="user@agribot.com")
    otp: str = Field(..., min_length=6, max_length=6, example="A7B2X9")

# --- 5. GANTI EMAIL ---
class ChangeEmailSchema(BaseModel): 
    token: str = Field(..., example="change-email-token-dari-verify-otp")
    new_email: EmailStr = Field(..., example="user@agribot.com")

# --- 6. GANTI PASSWORD ---
class ResetPasswordSchema(BaseModel):
    token: str = Field(..., example="reset-token-dari-verify-otp")
    new_password: str = Field(..., min_length=8, example="passwordbaru123")
    confirm_password: str = Field(..., min_length=8, example="passwordbaru123")

    @model_validator(mode="after")
    def passwords_match(self):
        if self.new_password != self.confirm_password:
            raise ValueError("Konfirmasi password tidak cocok.")
        return self
    
class RefreshTokenSchema(BaseModel):
    """Dipakai di endpoint POST /users/refresh-token"""
    refresh_token: str

class BulkLogoutSchema(BaseModel):
    """Dipakai di endpoint POST /users/sessions/logout-selected"""
    session_ids: List[int]

    @field_validator("session_ids")
    @classmethod
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("session_ids tidak boleh kosong.")
        if len(v) != len(set(v)):
            raise ValueError("session_ids tidak boleh mengandung duplikat.")
        return v