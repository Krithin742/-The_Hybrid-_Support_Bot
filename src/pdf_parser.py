"""
PDF Parser Module
Extracts text chunks with metadata (chapter, page) from technical PDFs
"""

import pdfplumber
import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a text chunk with associated metadata"""
    text: str
    metadata: Dict[str, any]


class PDFParser:
    """
    Parses PDF documents and extracts chunks with chapter/section metadata
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize parser with PDF file path
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        
    def extract_chunks_with_metadata(self) -> List[DocumentChunk]:
        """
        Parse PDF and extract chunks with chapter/section metadata
        
        Returns:
            List of DocumentChunk objects with text and metadata
        """
        chunks = []
        current_chapter = "Introduction"
        
        print(f"📖 Parsing PDF: {self.pdf_path}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📄 Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from page
                text = page.extract_text()
                
                if not text or len(text.strip()) < 50:
                    continue
                
                # Detect chapter headings
                detected_chapter = self._detect_chapter(text)
                if detected_chapter:
                    current_chapter = detected_chapter
                    print(f"  ✓ Page {page_num}: Found chapter '{current_chapter}'")
                
                # Split page into smaller chunks
                page_chunks = self._split_text(text)
                
                # Create DocumentChunk objects
                for chunk in page_chunks:
                    if len(chunk.strip()) > 100:  # Skip very short chunks
                        chunks.append(DocumentChunk(
                            text=chunk,
                            metadata={
                                "source": self.pdf_path,
                                "chapter": current_chapter,
                                "page": page_num
                            }
                        ))
        
        print(f"✅ Extracted {len(chunks)} chunks from {total_pages} pages")
        return chunks
    
    def _detect_chapter(self, text: str) -> str:
        """
        Detect chapter/section headings in text
        
        Args:
            text: Text to search for chapter headings
            
        Returns:
            Chapter name if found, None otherwise
        """
        # Common chapter heading patterns
        patterns = [
            r'^Chapter\s+\d+[:\.\s]+(.+)$',
            r'^CHAPTER\s+\d+[:\.\s]+(.+)$',
            r'^Section\s+\d+[:\.\s]+(.+)$',
            r'^SECTION\s+\d+[:\.\s]+(.+)$',
            r'^\d+\.\s+([A-Z][A-Za-z\s]{3,50})$',
            r'^([A-Z][A-Z\s]{3,50})$'  # ALL CAPS headers
        ]
        
        lines = text.split('\n')
        
        # Check first 5 lines for chapter headings
        for line in lines[:5]:
            line = line.strip()
            
            # Skip very long lines (not likely to be headings)
            if len(line) > 100 or len(line) < 3:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    chapter_name = match.group(1).strip()
                    # Clean up the chapter name
                    chapter_name = re.sub(r'\s+', ' ', chapter_name)
                    return chapter_name
        
        return None
    
    def _split_text(self, text: str, chunk_size: int = 800, 
                    overlap: int = 100) -> List[str]:
        """
        Split text into smaller chunks with overlap
        
        Args:
            text: Text to split
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences (rough)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-20:]) if len(words) > 20 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


# Test function
if __name__ == "__main__":
    # Test with a sample PDF
    parser = PDFParser("data/manual.pdf")
    chunks = parser.extract_chunks_with_metadata()
    
    print(f"\n{'='*60}")
    print(f"Sample chunk:")
    print(f"{'='*60}")
    if chunks:
        print(f"Text: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")