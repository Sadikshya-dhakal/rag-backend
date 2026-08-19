"""Raw text extraction from uploaded files (.pdf / .txt)."""
from __future__ import annotations

import io

import pdfplumber
from fastapi import HTTPException, UploadFile, status

SUPPORTED_TYPES = {"pdf", "txt"}


def infer_file_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Only .pdf and .txt are supported.",
        )
    return ext


def extract_text_from_pdf(raw_bytes: bytes) -> str:
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    text = "\n\n".join(pages).strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No extractable text found in PDF (it may be scanned/image-only).",
        )
    return text


def extract_text_from_txt(raw_bytes: bytes) -> str:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded .txt file is empty.")
    return text


async def extract_text(file: UploadFile) -> tuple[str, str]:
    """Returns (extracted_text, file_type)."""
    file_type = infer_file_type(file.filename or "")
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if file_type == "pdf":
        return extract_text_from_pdf(raw_bytes), file_type
    return extract_text_from_txt(raw_bytes), file_type
