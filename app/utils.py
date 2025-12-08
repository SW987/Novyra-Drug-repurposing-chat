import re
import PyPDF2
from pathlib import Path
from typing import NamedTuple, Optional


class DocumentInfo(NamedTuple):
    """Parsed document information from filename."""
    drug_id: str
    doc_id: str
    doc_title: str
    file_path: str


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

    # Split by underscores
    parts = name.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}. Expected: drug_repurposing_source_id.pdf")

    drug_id = parts[0]
    # Remove 'repurposing' from the middle
    if 'repurposing' in parts:
        repurposing_idx = parts.index('repurposing')
        source_id = '_'.join(parts[repurposing_idx + 1:])
    else:
        source_id = '_'.join(parts[1:])

    # Create human-readable title
    doc_title = f"{drug_id.title()} Repurposing {source_id}"

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
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text.strip()

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")


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
