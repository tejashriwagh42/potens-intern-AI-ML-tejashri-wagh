import os
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF
import docx
import pandas as pd

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a document based on its extension.
    Supported types: .pdf, .docx, .txt, .md, .csv
    
    Returns a list of dictionaries with extracted text and page numbers:
        [{"text": text_content, "page_number": page_num, "file_name": name, "file_type": ext}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()
    
    logger.info(f"Extracting text from {file_name} (type: {ext})")
    
    pages = []
    
    try:
        if ext == ".pdf":
            # Extract PDF page by page using PyMuPDF
            doc = fitz.open(file_path)
            for page_idx, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append({
                        "text": text,
                        "page_number": page_idx + 1,
                        "file_name": file_name,
                        "file_type": "pdf"
                    })
            doc.close()
            
        elif ext == ".docx":
            # Extract docx content
            doc = docx.Document(file_path)
            # Group text by paragraphs. Word documents don't have hard-coded page numbers in the XML,
            # so we'll group paragraphs into "pages" of about 2000 characters for consistency.
            current_text = []
            current_len = 0
            page_num = 1
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                current_text.append(text)
                current_len += len(text)
                
                # Treat every ~2000 chars as a new "page" or if we reach a threshold
                if current_len >= 2000:
                    pages.append({
                        "text": "\n".join(current_text),
                        "page_number": page_num,
                        "file_name": file_name,
                        "file_type": "docx"
                    })
                    current_text = []
                    current_len = 0
                    page_num += 1
                    
            if current_text:
                pages.append({
                    "text": "\n".join(current_text),
                    "page_number": page_num,
                    "file_name": file_name,
                    "file_type": "docx"
                })
                
        elif ext in [".txt", ".md"]:
            # Standard text / markdown
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            # If markdown is very long, we can split it by headings to simulate page-like splits
            # but for simplicity we treat it as page 1 or split by sections.
            # Let's split by double newlines or limit size
            lines = content.split("\n\n")
            current_text = []
            current_len = 0
            page_num = 1
            
            for line in lines:
                if not line.strip():
                    continue
                current_text.append(line)
                current_len += len(line)
                
                if current_len >= 2000:
                    pages.append({
                        "text": "\n\n".join(current_text),
                        "page_number": page_num,
                        "file_name": file_name,
                        "file_type": ext.replace(".", "")
                    })
                    current_text = []
                    current_len = 0
                    page_num += 1
                    
            if current_text:
                pages.append({
                    "text": "\n\n".join(current_text),
                    "page_number": page_num,
                    "file_name": file_name,
                    "file_type": ext.replace(".", "")
                })
                
        elif ext == ".csv":
            # Extract CSV row by row and format as text sentences
            df = pd.read_csv(file_path)
            rows_per_page = 20
            page_num = 1
            
            for i in range(0, len(df), rows_per_page):
                chunk_df = df.iloc[i : i + rows_per_page]
                row_texts = []
                for idx, row in chunk_df.iterrows():
                    # Format: Row 1: column1="value1", column2="value2"
                    items = [f'{col}="{val}"' for col, val in row.items() if pd.notna(val)]
                    row_texts.append(f"Record {idx + 1}: " + ", ".join(items))
                    
                pages.append({
                    "text": "\n".join(row_texts),
                    "page_number": page_num,
                    "file_name": file_name,
                    "file_type": "csv"
                })
                page_num += 1
                
        else:
            logger.warning(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")
            
    except Exception as e:
        logger.error(f"Error reading file {file_name}: {e}")
        raise e
        
    return pages
