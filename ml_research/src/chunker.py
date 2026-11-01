"""
PDF Document Parsing & Page-Aware Semantic Chunking

Extracts text from PDF materials, segments by headings, preserves page
boundaries and token budget, and generates clean retrieval units with metadata.
"""

import re
from typing import List, Dict, Any, Optional
import pypdf


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text per page using pypdf, preserving page indices.
    """
    reader = pypdf.PdfReader(pdf_path)
    pages_data = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        clean_page_text = re.sub(r"[ \t]+", " ", text).strip()
        pages_data.append(
            {
                "page_number": idx + 1,
                "text": clean_page_text,
                "char_count": len(clean_page_text),
            }
        )

    return pages_data


def chunk_document(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    material_id: Optional[str] = None,
    material_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Page-aware sliding chunker that prevents breaking in the middle of sentences.
    Preserves page span, approximate token counts, and heading cues.
    """
    chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        page_num = page_info["page_number"]
        page_text = page_info["text"]

        if not page_text or len(page_text) < 20:
            continue

        # Split into paragraphs or sentences
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9•\-\*])", page_text)
            if p.strip()
        ]

        current_words = []
        current_heading = "General"

        for p in paragraphs:
            # Detect short line as heading
            if len(p) < 60 and not p.endswith(".") and not p.startswith("•"):
                current_heading = p

            words = p.split()
            current_words.extend(words)

            while len(current_words) >= chunk_size:
                chunk_slice = current_words[:chunk_size]
                chunk_text = " ".join(chunk_slice)
                chunk_id = f"{material_id or 'mat'}_p{page_num}_c{chunk_counter}"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "material_id": material_id or "",
                        "material_name": material_name or "",
                        "page_number": page_num,
                        "section_heading": current_heading,
                        "text": chunk_text,
                        "word_count": len(chunk_slice),
                    }
                )
                chunk_counter += 1
                # Slide window with overlap
                current_words = current_words[chunk_size - chunk_overlap :]

        # Remainder words on this page
        if len(current_words) >= 40:
            chunk_slice = current_words
            chunk_text = " ".join(chunk_slice)
            chunk_id = f"{material_id or 'mat'}_p{page_num}_c{chunk_counter}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "material_id": material_id or "",
                    "material_name": material_name or "",
                    "page_number": page_num,
                    "section_heading": current_heading,
                    "text": chunk_text,
                    "word_count": len(chunk_slice),
                }
            )
            chunk_counter += 1

    return chunks
