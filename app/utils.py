import re
import logging
import warnings
from io import BytesIO

import PyPDF2
from PyPDF2.errors import PdfReadError, PdfReadWarning


_pypdf_logger = logging.getLogger("PyPDF2")
_pypdf_logger.setLevel(logging.ERROR)
_pypdf_logger.propagate = False
from pathlib import Path
from typing import NamedTuple, Optional


class DocumentInfo(NamedTuple):
    """Parsed document information from filename."""
    drug_id: str
    doc_id: str
    doc_title: str
    file_path: str


def _normalize_drug_id(drug_id: str) -> str:
    cleaned = drug_id.strip().lower()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def parse_filename(filename: str, drug_folder: str) -> DocumentInfo:
    """
    Parse filename with format: {drug_id}_repurposing_{source_id}.pdf

    Examples:
    - apomorphine_repurposing_PMC5995787.pdf
    - aspirin_repurposing_PMC11242460.pdf
    - insulin_repurposing_PMC11919260.pdf

    -> drug_id="apomorphine", doc_id="PMC5995787", doc_title="Apomorphine Repurposing PMC5995787"
    """
    # Remove file extension
    name = filename.rsplit('.', 1)[0]

    if "_repurposing_" in name:
        drug_id, source_id = name.split("_repurposing_", 1)
    else:
        parts = name.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {filename}. Expected: drug_repurposing_source_id.pdf")
        drug_id = parts[0]
        if 'repurposing' in parts:
            repurposing_idx = parts.index('repurposing')
            source_id = '_'.join(parts[repurposing_idx + 1:])
        else:
            source_id = '_'.join(parts[1:])

    drug_id = _normalize_drug_id(drug_id)

    display_name = drug_id.replace("_", " ").strip()
    # Create human-readable title
    doc_title = f"{display_name.title()} Repurposing {source_id}"

    return DocumentInfo(
        drug_id=drug_id.lower(),
        doc_id=source_id,
        doc_title=doc_title,
        file_path=f"{drug_folder}/{filename}"
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content

    Raises:
        Exception: If PDF cannot be processed
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PdfReadWarning)
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file, strict=False)
                if pdf_reader.is_encrypted:
                    pdf_reader.decrypt("")
                text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            for warning in caught:
                print(f"[WARN] {pdf_path}: {warning.message}")

            return text.strip()

    except PdfReadError as e:
        raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")


def extract_text_from_pdf_bytes(pdf_bytes: bytes, source_name: str = "<bytes>") -> str:
    """
    Extract text content from PDF bytes using PyPDF2.

    Args:
        pdf_bytes: Raw PDF bytes
        source_name: Label used in error messages

    Returns:
        Extracted text content
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PdfReadWarning)
            with BytesIO(pdf_bytes) as buffer:
                pdf_reader = PyPDF2.PdfReader(buffer, strict=False)
                if pdf_reader.is_encrypted:
                    pdf_reader.decrypt("")
                text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            for warning in caught:
                print(f"[WARN] {source_name}: {warning.message}")

            return text.strip()

    except PdfReadError as e:
        raise Exception(f"Failed to extract text from PDF {source_name}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF {source_name}: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters
    with `overlap` characters of overlap between chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_end = min(end + 100, len(text))
            sentence_end = text.rfind('.', end, search_end)
            if sentence_end != -1 and sentence_end > end - 100:
                end = sentence_end + 1
            else:
                # Look for word boundaries
                space_pos = text.rfind(' ', end - 50, end + 50)
                if space_pos != -1:
                    end = space_pos

        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position with overlap
        start = max(start + 1, end - overlap)

        # Prevent infinite loop
        if start >= len(text):
            break

    return chunks
