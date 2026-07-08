import re
import hashlib
from typing import List, Dict, Any

def detect_heading(text: str) -> str:
    """
    Detects if a sentence/line represents a section heading.
    Detects markdown-style headings (# Title) or uppercase lines.
    """
    text = text.strip()
    if not text:
        return ""
    
    # Check for Markdown headers (e.g., "# Introduction")
    if re.match(r"^#{1,6}\s+", text):
        return re.sub(r"^#{1,6}\s+", "", text).strip()
        
    # Check for capitalized short lines (e.g., "SECTION 1. REFUND POLICY")
    if len(text) < 80:
        # Matches "Section X", "Chapter Y", "1. Introduction"
        if re.match(r"^(Section|Chapter|Article|Policy|\d+\.\d+|\d+)\b", text, re.IGNORECASE):
            return text
        if text.isupper():
            return text
            
    return ""

def generate_document_id(file_name: str, content: str) -> str:
    """
    Generates a unique and deterministic document ID using MD5 hash.
    """
    hasher = hashlib.md5()
    hasher.update((file_name + content).encode("utf-8"))
    return f"doc_{hasher.hexdigest()[:12]}"

def create_chunks(pages: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Splits extracted pages into chunks of length `chunk_size` with `chunk_overlap` characters.
    Preserves sentence boundaries and prepends the active section heading for context preservation.
    
    Each chunk output looks like:
    {
        "text": chunk_text,
        "metadata": {
            "source_file": str,
            "page_number": int,
            "chunk_number": int,
            "document_id": str,
            "chunk_id": str
        }
    }
    """
    if not pages:
        return []
        
    # Generate unique doc_id based on total text contents
    full_text = "".join([p["text"] for p in pages])
    file_name = pages[0]["file_name"]
    doc_id = generate_document_id(file_name, full_text)
    
    chunks = []
    chunk_counter = 1
    
    for page in pages:
        text = page["text"]
        page_num = page["page_number"]
        
        # Simple sentence splitter
        # We split by common punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        buffer = []
        buffer_len = 0
        current_heading = ""
        
        idx = 0
        while idx < len(sentences):
            sentence = sentences[idx].strip()
            if not sentence:
                idx += 1
                continue
                
            # Check for a heading update
            heading_detected = detect_heading(sentence)
            if heading_detected:
                current_heading = heading_detected
                
            sentence_len = len(sentence)
            
            # If a single sentence exceeds the chunk_size, we force split it
            if sentence_len > chunk_size:
                # Flush the current buffer first if there is content
                if buffer:
                    chunk_text = " ".join(buffer)
                    if current_heading:
                        chunk_text = f"[Section: {current_heading}]\n{chunk_text}"
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source_file": file_name,
                            "page_number": page_num,
                            "chunk_number": chunk_counter,
                            "document_id": doc_id,
                            "chunk_id": f"{doc_id}_chunk_{chunk_counter}"
                        }
                    })
                    chunk_counter += 1
                    buffer = []
                    buffer_len = 0
                
                # Split large sentence by characters
                sub_idx = 0
                while sub_idx < sentence_len:
                    part = sentence[sub_idx : sub_idx + chunk_size]
                    if current_heading:
                        part = f"[Section: {current_heading}]\n{part}"
                    chunks.append({
                        "text": part,
                        "metadata": {
                            "source_file": file_name,
                            "page_number": page_num,
                            "chunk_number": chunk_counter,
                            "document_id": doc_id,
                            "chunk_id": f"{doc_id}_chunk_{chunk_counter}"
                        }
                    })
                    chunk_counter += 1
                    sub_idx += (chunk_size - chunk_overlap)
                
                idx += 1
                continue
            
            # Normal sentence accumulation
            if buffer_len + sentence_len + len(buffer) > chunk_size:
                # Emit current buffer
                chunk_text = " ".join(buffer)
                if current_heading:
                    chunk_text = f"[Section: {current_heading}]\n{chunk_text}"
                    
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": file_name,
                        "page_number": page_num,
                        "chunk_number": chunk_counter,
                        "document_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{chunk_counter}"
                    }
                })
                chunk_counter += 1
                
                # Setup overlap: backtrack buffer sentences to get ~150 chars of overlap
                overlap_buffer = []
                overlap_len = 0
                for prev_sent in reversed(buffer):
                    if overlap_len + len(prev_sent) + len(overlap_buffer) <= chunk_overlap:
                        overlap_buffer.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break
                
                buffer = overlap_buffer
                buffer_len = overlap_len
                
            buffer.append(sentence)
            buffer_len += sentence_len
            idx += 1
            
        # Emit final remaining buffer for the page
        if buffer:
            chunk_text = " ".join(buffer)
            if current_heading:
                chunk_text = f"[Section: {current_heading}]\n{chunk_text}"
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": file_name,
                    "page_number": page_num,
                    "chunk_number": chunk_counter,
                    "document_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_counter}"
                }
            })
            chunk_counter += 1
            
    return chunks
