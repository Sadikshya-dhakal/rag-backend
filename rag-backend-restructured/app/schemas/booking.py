from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr


class InterviewBookingCreate(BaseModel):
    session_id: str
    name: str
    email: EmailStr
    date: str  # ISO date, e.g. "2026-08-25"
    time: str  # "HH:MM", e.g. "14:30"


class InterviewBookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    name: str
    email: str
    interview_date: str
    interview_time: str
    status: str
    created_at: datetime
