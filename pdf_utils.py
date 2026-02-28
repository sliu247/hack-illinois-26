import fitz  # PyMuPDF
import re


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a text-based PDF.
    Raises error if PDF appears unreadable or scanned.
    """
    doc = fitz.open(file_path)

    if doc.page_count > 20:
        raise ValueError("PDF too long. Max 20 pages allowed.")

    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    if len(full_text.strip()) < 500:
        raise ValueError("PDF appears to be scanned or unreadable.")

    return full_text


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract raw text from a text-based PDF using raw bytes.
    Raises error if PDF appears unreadable or scanned.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if doc.page_count > 20:
        raise ValueError("PDF too long. Max 20 pages allowed.")

    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    if len(full_text.strip()) < 500:
        raise ValueError("PDF appears to be scanned or unreadable.")

    return full_text


def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - Remove excessive newlines
    - Remove simple page numbers
    - Remove references section
    """
    # Normalize newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove standalone page numbers
    text = re.sub(r"\n\d+\n", "\n", text)

    # Cut off references section (basic heuristic)
    if "References" in text:
        text = text.split("References")[0]

    return text.strip()


def chunk_text_by_paragraph(text: str, max_chars: int = 4000) -> list[str]:
    """
    Chunk text by paragraph without cutting sentences mid-way if possible.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_pdf(file_path: str) -> list[str]:
    """
    Main function your teammates will call.
    Returns cleaned text chunks ready for LLM input.
    """
    raw_text = extract_text_from_pdf(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text_by_paragraph(cleaned_text)

    if not chunks:
        raise ValueError("No valid text chunks extracted.")

    return chunks


def process_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """
    Process PDF from raw bytes.
    """
    raw_text = extract_text_from_bytes(pdf_bytes)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text_by_paragraph(cleaned_text)

    if not chunks:
        raise ValueError("No valid text chunks extracted.")

    return chunks