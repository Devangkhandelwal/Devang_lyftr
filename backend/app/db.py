"""SQLite persistence layer.

SQLite is the only storage engine used by this project: chat history and
session system prompts live in a local SQLite file. The schema is created
automatically on application startup.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS messages (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id  TEXT NOT NULL,
      role        TEXT NOT NULL,
      content     TEXT NOT NULL,
      metadata    TEXT,
      created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)",
    """
    CREATE TABLE IF NOT EXISTS sessions (
      session_id    TEXT PRIMARY KEY,
      system_prompt TEXT,
      created_at    TEXT NOT NULL
    )
    """,
]


def utc_now_iso() -> str:
    """Current server time as an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def resolve_db_path(database_url: str) -> Path:
    """Turn a DATABASE_URL into a filesystem path.

    Only SQLite URLs are supported. Four slashes means an absolute path, so
    sqlite:////data/app.db points at /data/app.db, while sqlite:///./app.db
    is relative to the process working directory.
    """
    url = database_url.strip()
    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw_path = url[len(prefix):]
    elif url.startswith("sqlite://"):
        raw_path = url[len("sqlite://"):]
    else:
        # Allow a bare filesystem path so the app is easy to run from a shell.
        raw_path = url
    if not raw_path:
        raise ValueError("Could not read a SQLite file path from DATABASE_URL")
    return Path(raw_path)


class Database:
    """Thin wrapper around sqlite3 using one short lived connection per call."""

    def __init__(self, database_url: str) -> None:
        self.path = resolve_db_path(database_url)

    def ensure_parent_dir(self) -> None:
        """Create the directory that holds the database file if it is missing."""
        parent = self.path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that returns rows as mappings."""
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def init_schema(self) -> None:
        """Create the tables and the index if they do not exist yet."""
        self.ensure_parent_dir()
        with self.connect() as connection:
            # WAL keeps readers from blocking the writer, which matters once
            # the API serves concurrent chat requests.
            connection.execute("PRAGMA journal_mode=WAL")
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.commit()

    def ping(self) -> bool:
        """Return True when the database answers a trivial query."""
        with self.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    # ------------------------------ sessions ------------------------------

    def get_system_prompt(self, session_id: str) -> str | None:
        """Return the stored system prompt for a session, or None."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["system_prompt"] if row is not None else None

    def session_exists(self, session_id: str) -> bool:
        """True when the session is known, either as a session or a message."""
        with self.connect() as connection:
            session_row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if session_row is not None:
                return True
            message_row = connection.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return message_row is not None

    def save_session_prompt(self, session_id: str, system_prompt: str | None) -> None:
        """Create or update the session row holding the system prompt.

        The system prompt is stored here only. It is never written to the
        messages table, so it can never leak into the history response.
        """
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO sessions (session_id, system_prompt, created_at)"
                    " VALUES (?, ?, ?)",
                    (session_id, system_prompt, utc_now_iso()),
                )
            elif system_prompt is not None:
                connection.execute(
                    "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                    (system_prompt, session_id),
                )
            connection.commit()

    # ------------------------------ messages ------------------------------

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return the stored history for a session, oldest first."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT role, content, metadata FROM messages"
                " WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

        history: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["metadata"]:
                try:
                    item["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    # A corrupt metadata blob must not break the endpoint.
                    item["metadata"] = None
            history.append(item)
        return history

    def count_messages(self, session_id: str) -> int:
        """Number of stored messages for a session. Used by the tests."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["total"])

    def save_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        assistant_metadata: dict[str, Any],
        system_prompt: str | None = None,
    ) -> None:
        """Insert the session row and both message rows in one transaction.

        This runs only after a successful generation, so a failed request
        never leaves a partial exchange, or even a bare session, behind.
        """
        created_at = utc_now_iso()
        metadata_json = json.dumps(assistant_metadata, separators=(",", ":"))
        with self.connect() as connection:
            try:
                connection.execute("BEGIN")
                existing = connection.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO sessions (session_id, system_prompt, created_at)"
                        " VALUES (?, ?, ?)",
                        (session_id, system_prompt, created_at),
                    )
                elif system_prompt is not None:
                    connection.execute(
                        "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                        (system_prompt, session_id),
                    )
                connection.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (session_id, "user", user_content, None, created_at),
                )
                connection.execute(
                    "INSERT INTO messages (session_id, role, content, metadata, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (session_id, "assistant", assistant_content, metadata_json, created_at),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
