import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ChromaStore:
    def __init__(
        self,
        db_path: str = "chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "document_qa"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Ensure db directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize persistent Client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model locally
        logger.info(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for scores
        )
        
    def check_document_exists(self, file_name: str) -> bool:
        """
        Checks if a file with the given name has already been ingested.
        """
        results = self.collection.get(
            where={"source_file": file_name},
            limit=1
        )
        return len(results["ids"]) > 0

    def add_documents(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Adds text chunks to the Chroma DB.
        Expects a list of dicts:
        {
            "text": str,
            "metadata": {
                "source_file": str,
                "page_number": int,
                "chunk_number": int,
                "document_id": str,
                "chunk_id": str
            }
        }
        """
        if not chunks:
            return ""
            
        file_name = chunks[0]["metadata"]["source_file"]
        doc_id = chunks[0]["metadata"]["document_id"]
        
        if self.check_document_exists(file_name):
            logger.warning(f"Document {file_name} already exists. Skipping ingestion to avoid duplicates.")
            return doc_id
            
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk["metadata"]["chunk_id"])
            texts.append(chunk["text"])
            # ChromaDB metadatas values must be string, int, float, or bool
            metadatas.append(chunk["metadata"])
            
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks of {file_name}...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True).tolist()
        
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Successfully ingested {len(texts)} chunks for {file_name} (ID: {doc_id}).")
        return doc_id

    def query_collection(self, query_text: str, top_k: int = 10, where_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Queries the vector store for the top_k most similar chunks.
        """
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        formatted_results = []
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return formatted_results
            
        # Chroma query returns a list of lists (since we query list of embeddings)
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
        
        for i in range(len(ids)):
            # Convert cosine distance to similarity score
            # ChromaDB with 'cosine' space returns (1 - cosine_similarity), i.e., cosine distance.
            # So similarity = 1 - distance
            distance = distances[i]
            similarity = max(0.0, min(1.0, 1.0 - distance))
            
            formatted_results.append({
                "chunk_id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "similarity_score": similarity
            })
            
        return formatted_results

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lists all unique documents indexed in the vector store.
        """
        # Fetch all metadatas to consolidate unique docs
        all_data = self.collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        seen_docs = {}
        for meta in metadatas:
            doc_id = meta.get("document_id")
            source_file = meta.get("source_file")
            file_type = meta.get("file_type", "")
            
            if doc_id and doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "document_id": doc_id,
                    "file_name": source_file,
                    "file_type": file_type,
                    "chunks_count": 0
                }
            if doc_id:
                seen_docs[doc_id]["chunks_count"] += 1
                
        return list(seen_docs.values())

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes all chunks associated with a document_id.
        """
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted document {document_id} from vector store.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
            
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Fetches all chunks and their texts for a specific document.
        """
        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )
        chunks = []
        for idx in range(len(results["ids"])):
            chunks.append({
                "chunk_id": results["ids"][idx],
                "text": results["documents"][idx],
                "metadata": results["metadatas"][idx]
            })
        return chunks
