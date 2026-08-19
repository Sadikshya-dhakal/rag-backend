"""Data-access layer for interview bookings."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.booking import InterviewBooking


class BookingRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, booking: InterviewBooking) -> InterviewBooking:
        self._db.add(booking)
        self._db.commit()
        self._db.refresh(booking)
        return booking

    def get(self, booking_id: str) -> InterviewBooking | None:
        return self._db.get(InterviewBooking, booking_id)

    def list_all(self) -> list[InterviewBooking]:
        return list(
            self._db.execute(select(InterviewBooking).order_by(InterviewBooking.created_at.desc())).scalars().all()
        )
