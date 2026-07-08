import os
import json
import logging
from typing import List, Dict, Any, Tuple
from langdetect import detect
import google.generativeai as genai
from groq import Groq

from vectorstore.chroma_store import ChromaStore
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# Map code to readable language name
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French"
}

class RetrievalPipeline:
    def __init__(self, chroma_store: ChromaStore, reranker: Reranker):
        self.store = chroma_store
        self.reranker = reranker
        
        # Load configs
        self.llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        self.reranking_enabled = os.environ.get("RERANKING_ENABLED", "true").lower() == "true"
        self.top_k = int(os.environ.get("TOP_K", "10"))
        
        # Configure Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            # Support configuring GEMINI_MODEL via env, defaulting to gemini-2.5-flash
            gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            self.gemini_model = genai.GenerativeModel(gemini_model_name)
        else:
            self.gemini_model = None
            logger.warning("GEMINI_API_KEY is not set in environment.")

        # Configure Groq
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            self.groq_client = None
            logger.warning("GROQ_API_KEY is not set in environment.")

    def _call_llm(self, prompt: str, json_mode: bool = False) -> str:
        """
        Helper method to call LLM (Gemini or Groq) based on configuration.
        """
        import time
        for attempt in range(5):
            try:
                if self.llm_provider == "gemini":
                    if not self.gemini_model:
                        raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY in .env.")
                    
                    generation_config = {}
                    if json_mode:
                        generation_config["response_mime_type"] = "application/json"
                        
                    response = self.gemini_model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    return response.text.strip()
                    
                elif self.llm_provider == "groq":
                    if not self.groq_client:
                        raise ValueError("Groq client is not initialized. Please set GROQ_API_KEY in .env.")
                        
                    response_format = {"type": "json_object"} if json_mode else None
                    
                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),  # High-performance Llama model on Groq
                        response_format=response_format,
                        temperature=0.0
                    )
                    return chat_completion.choices[0].message.content.strip()
                else:
                    raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
            except Exception as e:
                err_msg = str(e).lower()
                if "quota" in err_msg or "rate limit" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                    sleep_time = 15 * (attempt + 1)
                    logger.warning(f"Rate limit hit in _call_llm. Sleeping for {sleep_time}s before retrying. Error: {e}")
                    time.sleep(sleep_time)
                else:
                    raise e
        raise RuntimeError("Failed to call LLM after 5 attempts due to rate limits.")

    def detect_and_translate_query(self, query: str) -> Tuple[str, str]:
        """
        Detects query language and translates it to English if needed.
        Returns: (detected_lang_code, translated_query)
        """
        try:
            detected_code = detect(query)
            # Default to English if not one of our target multilingual options
            if detected_code not in LANG_MAP:
                detected_code = "en"
        except Exception:
            detected_code = "en"
            
        if detected_code == "en":
            return "en", query
            
        logger.info(f"Query detected language: {LANG_MAP.get(detected_code)} ({detected_code}). Translating to English.")
        
        prompt = (
            f"Translate the following user question from {LANG_MAP.get(detected_code)} to English. "
            f"Return ONLY the translated text, do not include any introductory remarks, quotes, or formatting.\n\n"
            f"Question: {query}"
        )
        
        try:
            translated = self._call_llm(prompt, json_mode=False)
            logger.info(f"Translated query: '{translated}'")
            return detected_code, translated
        except Exception as e:
            logger.error(f"Error during query translation: {e}. Falling back to original query.")
            return detected_code, query

    def translate_response(self, text: str, target_lang: str) -> str:
        """
        Translates answer text from English back to target language.
        """
        if target_lang == "en" or target_lang not in LANG_MAP:
            return text
            
        logger.info(f"Translating response to {LANG_MAP.get(target_lang)}")
        
        prompt = (
            f"Translate the following English response into fluent, natural {LANG_MAP.get(target_lang)}. "
            f"Keep any numbers, names, or file references unchanged. "
            f"Return ONLY the translated text, do not include comments, quotes, or additional formatting.\n\n"
            f"Text to translate: {text}"
        )
        
        try:
            translated = self._call_llm(prompt, json_mode=False)
            return translated
        except Exception as e:
            logger.error(f"Error during response translation: {e}. Returning English text.")
            return text

    def compute_confidence_score(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        cited_chunk_ids: List[str],
        reranked: bool
    ) -> float:
        """
        Confidence formula:
        Score = 0.5 * S_similarity + 0.3 * S_rerank + 0.2 * S_coverage
        
        - S_similarity: Avg similarity score of top-3 retrieved chunks (scaled to [0,1])
        - S_rerank: Normalized rerank score of top chunk (or top similarity score if not reranked)
        - S_coverage: Cited chunks count / Retrieved chunks count
        """
        if not retrieved_chunks:
            return 0.0
            
        # 1. Similarity score (scaled 0-1)
        sim_scores = [c.get("similarity_score", 0.0) for c in retrieved_chunks]
        top_3_sims = sim_scores[:3]
        s_similarity = sum(top_3_sims) / len(top_3_sims) if top_3_sims else 0.0
        
        # 2. Rerank score
        if reranked:
            s_rerank = retrieved_chunks[0].get("rerank_score", s_similarity)
        else:
            s_rerank = retrieved_chunks[0].get("similarity_score", s_similarity)
            
        # 3. Coverage score
        retrieved_ids = {c["chunk_id"] for c in retrieved_chunks}
        # Filter citations to only count those that were actually part of retrieved chunks
        valid_cited = [cid for cid in cited_chunk_ids if cid in retrieved_ids]
        s_coverage = len(set(valid_cited)) / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        # Weighted sum
        score = (0.5 * s_similarity) + (0.3 * s_rerank) + (0.2 * s_coverage)
        return round(max(0.0, min(1.0, score)), 2)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main Q&A workflow.
        1. Language detection & translation
        2. Vector store retrieval (Top-k=10)
        3. Optional Reranking
        4. LLM Generation with citations
        5. Confidence scoring
        6. Translation of final answer
        """
        # Step 1: Detect and translate query
        lang_code, eng_question = self.detect_and_translate_query(question)
        
        # Step 2: Retrieve chunks (always query using the English representation)
        logger.info(f"Retrieving from ChromaDB for English query: '{eng_question}'")
        retrieved = self.store.query_collection(eng_question, top_k=self.top_k)
        
        if not retrieved:
            return {
                "answer": "The uploaded documents do not contain enough information to answer this question.",
                "confidence_score": 0.0,
                "warning": None,
                "citations": []
            }
            
        # Step 3: Rerank
        reranked = False
        if self.reranking_enabled and self.reranker:
            try:
                retrieved = self.reranker.rerank(eng_question, retrieved)
                reranked = True
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Proceeding with base vector retrieval order.")
                
        # Step 4: Build prompt and invoke LLM
        context_str = ""
        for idx, chunk in enumerate(retrieved):
            meta = chunk["metadata"]
            context_str += (
                f"[{idx + 1}] Chunk ID: {chunk['chunk_id']}\n"
                f"Source File: {meta.get('source_file')}\n"
                f"Page: {meta.get('page_number')}\n"
                f"Text:\n{chunk['text']}\n"
                f"----------------------------------------\n"
            )
            
        prompt = (
            f"You are a factual, multilingual QA assistant.\n"
            f"Analyze the context chunks below to answer the user's question. "
            f"Every fact in your answer MUST have a citation referencing the chunk index.\n\n"
            f"Context Chunks:\n{context_str}\n"
            f"User Question: {eng_question}\n\n"
            f"Instructions:\n"
            f"1. Respond ONLY in JSON matching this exact structure:\n"
            f"{{\n"
            f"  \"answer\": \"Detailed English answer strictly based on the context.\",\n"
            f"  \"citations\": [\n"
            f"    {{\n"
            f"      \"source_file\": \"filename.pdf\",\n"
            f"      \"page\": 1,\n"
            f"      \"chunk_id\": \"doc_xyz_chunk_1\",\n"
            f"      \"snippet\": \"exact quote from the context that supports this point\"\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"2. If the context does not contain enough information, or is completely irrelevant, return:\n"
            f"{{\n"
            f"  \"answer\": \"The uploaded documents do not contain enough information to answer this question.\",\n"
            f"  \"citations\": []\n"
            f"}}\n\n"
            f"3. Never invent facts. Do not assume or extrapolate beyond the provided text. Make citations precise."
        )
        
        try:
            llm_response = self._call_llm(prompt, json_mode=True)
            data = json.loads(llm_response)
        except Exception as e:
            logger.error(f"Error parsing LLM response or querying API: {e}")
            return {
                "answer": "Error generating answer. Please check LLM API logs.",
                "confidence_score": 0.0,
                "warning": None,
                "citations": []
            }
            
        answer_eng = data.get("answer", "The uploaded documents do not contain enough information to answer this question.")
        citations = data.get("citations", [])
        
        # Check if the answer is negative (insufficient information)
        is_insufficient = "do not contain enough information" in answer_eng.lower()
        
        # Step 5: Confidence score
        cited_chunk_ids = [c.get("chunk_id") for c in citations if c.get("chunk_id")]
        
        if is_insufficient:
            confidence = 0.0
            citations = []
        else:
            confidence = self.compute_confidence_score(retrieved, cited_chunk_ids, reranked)
            
        # Step 6: Translate back if query was multilingual
        if lang_code != "en" and not is_insufficient:
            answer_final = self.translate_response(answer_eng, lang_code)
            # Also translate snippets to align language, keeping filenames unchanged
            for cit in citations:
                if cit.get("snippet"):
                    cit["snippet"] = self.translate_response(cit["snippet"], lang_code)
        else:
            answer_final = answer_eng
            
        # Trigger human review warning
        warning = None
        if confidence < 0.50 and not is_insufficient:
            warning = "Low confidence answer. Please verify manually."
            
        return {
            "answer": answer_final,
            "confidence_score": confidence,
            "warning": warning,
            "citations": citations
        }
