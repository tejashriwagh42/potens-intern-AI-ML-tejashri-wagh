import os
import json
import logging
from typing import Dict, Any
import google.generativeai as genai
from groq import Groq
from vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

class ContradictionDetector:
    def __init__(self, chroma_store: ChromaStore):
        self.store = chroma_store
        
        self.llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        
        # Configure Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
        else:
            self.gemini_model = None
            
        # Configure Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            self.groq_client = None

    def _call_llm(self, prompt: str) -> str:
        """
        Helper method to call configured LLM in JSON mode.
        """
        if self.llm_provider == "gemini":
            if not self.gemini_model:
                raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY in .env.")
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text.strip()
            
        elif self.llm_provider == "groq":
            if not self.groq_client:
                raise ValueError("Groq client is not initialized. Please set GROQ_API_KEY in .env.")
                
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return chat_completion.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def detect_contradiction(self, doc1_id: str, doc2_id: str, topic: str) -> Dict[str, Any]:
        """
        Retrieves chunks from both doc1 and doc2 matching the topic.
        Asks LLM to compare for contradictions.
        """
        logger.info(f"Detecting contradictions between {doc1_id} and {doc2_id} on topic: '{topic}'")
        
        # 1. Retrieve chunks for doc1
        doc1_chunks = self.store.query_collection(
            query_text=topic,
            top_k=5,
            where_filter={"document_id": doc1_id}
        )
        
        # 2. Retrieve chunks for doc2
        doc2_chunks = self.store.query_collection(
            query_text=topic,
            top_k=5,
            where_filter={"document_id": doc2_id}
        )
        
        # If either doc has no relevant chunks, we cannot analyze conflicts
        if not doc1_chunks or not doc2_chunks:
            doc1_name = doc1_chunks[0]["metadata"]["source_file"] if doc1_chunks else doc1_id
            doc2_name = doc2_chunks[0]["metadata"]["source_file"] if doc2_chunks else doc2_id
            
            reason = f"Insufficient information found in one or both documents. "
            if not doc1_chunks:
                reason += f"No topic-related content was found in document '{doc1_name}'. "
            if not doc2_chunks:
                reason += f"No topic-related content was found in document '{doc2_name}'."
                
            return {
                "conflict": "unknown",
                "reasoning": reason.strip()
            }
            
        # Format contexts for LLM
        doc1_name = doc1_chunks[0]["metadata"]["source_file"]
        doc2_name = doc2_chunks[0]["metadata"]["source_file"]
        
        context_doc1 = "\n".join([f"- Page {c['metadata']['page_number']}: {c['text']}" for c in doc1_chunks])
        context_doc2 = "\n".join([f"- Page {c['metadata']['page_number']}: {c['text']}" for c in doc2_chunks])
        
        prompt = (
            f"You are a factual contradiction detector.\n"
            f"Compare statement(s) about the topic: '{topic}' from two documents:\n\n"
            f"Document 1 (File: {doc1_name}):\n{context_doc1}\n\n"
            f"Document 2 (File: {doc2_name}):\n{context_doc2}\n\n"
            f"Tasks:\n"
            f"1. Analyze if there are statements in Document 1 and Document 2 about '{topic}' that contradict or conflict with each other.\n"
            f"2. Note: A conflict means they assert opposing or incompatible claims (e.g. 30-day refund limit vs no refunds allowed).\n"
            f"3. Return ONLY a JSON object with this exact format:\n"
            f"{{\n"
            f"  \"conflict\": true / false / \"unknown\",\n"
            f"  \"reasoning\": \"Detailed analysis. If conflict is true, explain exactly what the conflict is, citing page numbers and file names. If conflict is false, state that they are consistent or don't contradict. If conflict is 'unknown', explain why the evidence is insufficient to find a conflict.\"\n"
            f"}}\n\n"
            f"Do not include any notes, formatting backticks, or other text outside the JSON."
        )
        
        try:
            llm_response = self._call_llm(prompt)
            result = json.loads(llm_response)
            
            # Standardize conflict field to True, False or "unknown"
            conflict_val = result.get("conflict")
            if isinstance(conflict_val, str):
                conflict_lower = conflict_val.lower()
                if conflict_lower == "true":
                    result["conflict"] = True
                elif conflict_lower == "false":
                    result["conflict"] = False
                else:
                    result["conflict"] = "unknown"
            elif isinstance(conflict_val, bool):
                pass
            else:
                result["conflict"] = "unknown"
                
            return result
            
        except Exception as e:
            logger.error(f"Error during contradiction detection LLM call: {e}")
            return {
                "conflict": "unknown",
                "reasoning": f"An error occurred during contradiction analysis: {str(e)}"
            }
