from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_booking_repository
from app.models.booking import InterviewBooking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import InterviewBookingCreate, InterviewBookingResponse
from app.services.booking_service import create_booking

router = APIRouter(prefix="/bookings", tags=["Interview Bookings"])


@router.post("", response_model=InterviewBookingResponse, status_code=status.HTTP_201_CREATED)
def create_booking_direct(
    booking_in: InterviewBookingCreate, repository: BookingRepository = Depends(get_booking_repository)
) -> InterviewBooking:
    """Direct booking creation, bypassing the chat/LLM flow (useful for
    programmatic clients that already have all the details)."""
    return create_booking(repository, booking_in)


@router.get("", response_model=list[InterviewBookingResponse])
def list_bookings(repository: BookingRepository = Depends(get_booking_repository)) -> list[InterviewBooking]:
    return repository.list_all()


@router.get("/{booking_id}", response_model=InterviewBookingResponse)
def get_booking(
    booking_id: str, repository: BookingRepository = Depends(get_booking_repository)
) -> InterviewBooking:
    booking = repository.get(booking_id)
    if booking is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Booking not found.")
    return booking
