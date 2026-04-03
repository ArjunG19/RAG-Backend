"""SQLite-backed document persistence service."""

from __future__ import annotations

import aiosqlite

from app.models.internal import DocumentRecord

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    source TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    file_data BLOB NOT NULL,
    uploaded_at TEXT NOT NULL
)
"""


class DocumentStore:
    """Manages document persistence in a SQLite database.

    Provides async CRUD operations for DocumentRecord instances
    using aiosqlite for non-blocking database access.
    """

    def __init__(self, db_url: str) -> None:
        if db_url == "sqlite:///:memory:" or db_url == ":memory:":
            self._db_path = ":memory:"
        elif db_url.startswith("sqlite:///"):
            self._db_path = db_url[len("sqlite:///"):]
        else:
            self._db_path = db_url
        self._connection: aiosqlite.Connection | None = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Return a reusable connection (required for :memory: databases)."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self._db_path)
            self._connection.row_factory = aiosqlite.Row
        return self._connection

    async def init_db(self) -> None:
        """Create the documents table if it does not already exist."""
        db = await self._get_connection()
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()

    async def save(self, record: DocumentRecord) -> None:
        """Insert a DocumentRecord into the database."""
        db = await self._get_connection()
        await db.execute(
            """
            INSERT INTO documents
                (id, filename, content_type, source, file_size, chunk_count, file_data, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.filename,
                record.content_type,
                record.source,
                record.file_size,
                record.chunk_count,
                record.file_data,
                record.uploaded_at,
            ),
        )
        await db.commit()

    async def get(self, document_id: str) -> DocumentRecord | None:
        """Fetch a document by ID, returning all fields including file_data."""
        db = await self._get_connection()
        cursor = await db.execute(
            "SELECT id, filename, content_type, source, file_size, "
            "chunk_count, file_data, uploaded_at "
            "FROM documents WHERE id = ?",
            (document_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            content_type=row["content_type"],
            source=row["source"],
            file_size=row["file_size"],
            chunk_count=row["chunk_count"],
            file_data=bytes(row["file_data"]),
            uploaded_at=row["uploaded_at"],
        )

    async def list_all(self) -> list[DocumentRecord]:
        """Return all records ordered by uploaded_at DESC with empty file_data."""
        db = await self._get_connection()
        cursor = await db.execute(
            "SELECT id, filename, content_type, source, file_size, "
            "chunk_count, uploaded_at "
            "FROM documents ORDER BY uploaded_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            DocumentRecord(
                id=row["id"],
                filename=row["filename"],
                content_type=row["content_type"],
                source=row["source"],
                file_size=row["file_size"],
                chunk_count=row["chunk_count"],
                file_data=b"",
                uploaded_at=row["uploaded_at"],
            )
            for row in rows
        ]

    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID. Returns True if a row was removed."""
        db = await self._get_connection()
        cursor = await db.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def update_chunk_count(self, document_id: str, chunk_count: int) -> None:
        """Update the chunk_count column for a given document."""
        db = await self._get_connection()
        await db.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (chunk_count, document_id),
        )
        await db.commit()

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
