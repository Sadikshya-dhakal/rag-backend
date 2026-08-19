"""Redis-backed short-term conversational memory.

Each session's rolling window of turns is stored as a Redis list under
`chat:{session_id}`, capped at `chat_memory_max_turns` and expiring after
`chat_memory_ttl_seconds` of inactivity. This is intentionally separate
from the durable `ChatMessage` SQL log: Redis holds the *working* context
used for prompting, and is cheap to trim/expire.

Booking state (in-progress slot filling) is stored the same way under a
separate key so multi-turn slot collection survives across requests.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import redis

from app.core.config import get_settings


class ChatMemory:
    def __init__(self) -> None:
        settings = get_settings()
        self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._max_turns = settings.chat_memory_max_turns
        self._ttl = settings.chat_memory_ttl_seconds

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:history:{session_id}"

    @staticmethod
    def _booking_key(session_id: str) -> str:
        return f"chat:booking:{session_id}"

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        key = self._history_key(session_id)
        entry = json.dumps({"role": role, "content": content})
        pipe = self._redis.pipeline()
        pipe.rpush(key, entry)
        pipe.ltrim(key, -self._max_turns, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        raw = self._redis.lrange(self._history_key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def clear_history(self, session_id: str) -> None:
        self._redis.delete(self._history_key(session_id))

    # --- Booking slot-filling state ---

    def get_booking_state(self, session_id: str) -> dict[str, Any]:
        raw = self._redis.get(self._booking_key(session_id))
        return json.loads(raw) if raw else {}

    def set_booking_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._redis.set(self._booking_key(session_id), json.dumps(state), ex=self._ttl)

    def clear_booking_state(self, session_id: str) -> None:
        self._redis.delete(self._booking_key(session_id))


@lru_cache
def get_chat_memory() -> ChatMemory:
    return ChatMemory()
