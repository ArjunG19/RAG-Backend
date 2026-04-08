"""SQLite-backed document persistence service."""

from __future__ import annotations

import aiosqlite

from app.models.internal import DocumentRecord

_CREATE_DOCUMENTS_TABLE_SQL = """
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

_CREATE_BM25_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bm25_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
)
"""

_CREATE_BM25_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_bm25_chunks_document_id
ON bm25_chunks (document_id)
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
            # Enable foreign key enforcement
            await self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    async def init_db(self) -> None:
        """Create tables if they do not already exist."""
        db = await self._get_connection()
        await db.execute(_CREATE_DOCUMENTS_TABLE_SQL)
        await db.execute(_CREATE_BM25_CHUNKS_TABLE_SQL)
        await db.execute(_CREATE_BM25_INDEX_SQL)
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
        """Delete a document and its BM25 chunks by ID (CASCADE handles chunks)."""
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

    # ------------------------------------------------------------------
    # BM25 chunk persistence
    # ------------------------------------------------------------------

    async def save_bm25_chunks(self, chunks: list) -> None:
        """Persist text chunks for BM25 index rebuilding on startup.

        Args:
            chunks: List of TextChunk objects with .text, .chunk_index,
                    and .metadata dict containing 'document_id' and 'source'.
        """
        db = await self._get_connection()
        rows = [
            (
                chunk.metadata.get("document_id", ""),
                chunk.chunk_index,
                chunk.metadata.get("source", ""),
                chunk.text,
            )
            for chunk in chunks
        ]
        await db.executemany(
            "INSERT INTO bm25_chunks (document_id, chunk_index, source, text) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        await db.commit()

    async def load_all_bm25_chunks(self) -> list[dict]:
        """Load all BM25 chunks ordered by document and chunk index.

        Returns:
            List of dicts with keys: document_id, chunk_index, source, text.
        """
        db = await self._get_connection()
        cursor = await db.execute(
            "SELECT document_id, chunk_index, source, text "
            "FROM bm25_chunks "
            "ORDER BY document_id, chunk_index"
        )
        rows = await cursor.fetchall()
        return [
            {
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "source": row["source"],
                "text": row["text"],
            }
            for row in rows
        ]

    async def delete_bm25_chunks_by_document(self, document_id: str) -> int:
        """Delete all BM25 chunks for a given document.

        Returns:
            Number of rows deleted.
        """
        db = await self._get_connection()
        cursor = await db.execute(
            "DELETE FROM bm25_chunks WHERE document_id = ?",
            (document_id,),
        )
        await db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None