"""File parsers for extracting text from uploaded documents.

Supports PDF, plain text, and image (PNG/JPEG via OCR) file types.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod

from PIL import Image
from pypdf import PdfReader
import pytesseract


class FileParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    async def parse(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from a file.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename for context.

        Returns:
            Extracted text content as a string.
        """
        ...


class PDFParser(FileParser):
    """Extract text from all pages of a PDF document using pypdf."""

    async def parse(self, file_bytes: bytes, filename: str) -> str:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)


class TextParser(FileParser):
    """Decode raw bytes to a UTF-8 string."""

    async def parse(self, file_bytes: bytes, filename: str) -> str:
        return file_bytes.decode("utf-8")


class ImageParser(FileParser):
    """Extract text from images (PNG/JPEG) via pytesseract OCR."""

    async def parse(self, file_bytes: bytes, filename: str) -> str:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)


# Mapping of MIME types to parser instances
_PARSER_MAP: dict[str, FileParser] = {
    "application/pdf": PDFParser(),
    "text/plain": TextParser(),
    "image/png": ImageParser(),
    "image/jpeg": ImageParser(),
}


class ParserFactory:
    """Factory that returns the appropriate parser for a given content type."""

    @staticmethod
    def get_parser(content_type: str) -> FileParser:
        """Return a FileParser for the given MIME content type.

        Args:
            content_type: MIME type of the uploaded file.

        Returns:
            A FileParser instance capable of handling the content type.

        Raises:
            ValueError: If the content type is not supported.
        """
        parser = _PARSER_MAP.get(content_type)
        if parser is None:
            supported = ", ".join(sorted(_PARSER_MAP.keys()))
            raise ValueError(
                f"Unsupported file type: {content_type}. "
                f"Supported: {supported}"
            )
        return parser
