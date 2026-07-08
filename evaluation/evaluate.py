import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple

# Ensure parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

from vectorstore.chroma_store import ChromaStore
from retrieval.reranker import Reranker
from retrieval.retriever import RetrievalPipeline
from ingestion.loader import extract_text_from_file
from ingestion.chunker import create_chunks

# Constants for sample documents (in case they don't exist yet)
SAMPLE_DOCS = {
    "refund_policy.txt": (
        "# Company Refund Policy\n\n"
        "Section 1: General Terms\n"
        "We want you to be completely satisfied with your purchase. Refunds are available for physical items within 30 days of the purchase date, provided the items are unused and in original packaging.\n\n"
        "Section 2: Digital Products\n"
        "Digital downloads and software licenses are non-refundable once the download link is accessed or the license key is activated. Subscriptions can be canceled at any time, but no partial refunds will be given.\n"
    ),
    "terms_of_service.txt": (
        "# Terms of Service\n\n"
        "Section 1: User Accounts\n"
        "Users are responsible for maintaining the confidentiality of their credentials. The company may suspend or terminate accounts immediately without prior notice if the user violates terms, engages in fraudulent behavior, or fails to pay fees.\n\n"
        "Section 2: Disputes and Support\n"
        "Users must contact the billing support team at billing@company.com within 60 days of the invoice date to dispute billing issues. Any dispute raised after 60 days will be considered invalid.\n"
    ),
    "privacy_policy.txt": (
        "# Privacy Policy\n\n"
        "Section 1: Data Protection\n"
        "Personal data is encrypted in transit using TLS 1.3 and at rest using AES-256 encryption. Access is restricted to authorized personnel under strict confidentiality agreements.\n\n"
        "Section 2: Data Sharing\n"
        "No, the company does not sell, rent, or lease personal information to third parties for marketing or advertising purposes. Data is only shared with trusted service providers essential for application operation.\n"
    ),
    "employee_handbook.txt": (
        "# Employee Handbook\n\n"
        "Section 1: Office Hours\n"
        "Standard working hours are 9:00 AM to 5:30 PM, Monday through Friday, including a mandatory 45-minute unpaid lunch break.\n\n"
        "Section 2: Paid Time Off (PTO)\n"
        "Full-time employees receive 15 days of paid annual vacation, accrued monthly at a rate of 1.25 days per month. Unused PTO can carry over up to a maximum of 5 days per year.\n"
    ),
    "travel_policy.txt": (
        "# Travel Expense Policy\n\n"
        "Section 1: Meal Allowances\n"
        "The maximum daily reimbursement limit for meals during domestic business travel is $75, requiring itemized receipts for purchases over $25.\n\n"
        "Section 2: Flights\n"
        "No, economy class must be booked for all flights under 4 hours. Business class is only permitted for flights exceeding 4 hours, subject to pre-approval by the department head.\n"
    )
}

def setup_evaluation_environment():
    """
    Ensures documents folder and sample files exist. Ingests them into database if empty.
    """
    docs_dir = os.environ.get("DOCUMENTS_DIR", "documents")
    db_path = os.environ.get("CHROMA_DB_PATH", "chroma_db")
    
    os.makedirs(docs_dir, exist_ok=True)
    store = ChromaStore(db_path=db_path)
    
    # Write sample files if not present on disk
    for file_name, content in SAMPLE_DOCS.items():
        file_path = os.path.join(docs_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Created sample document: {file_path}")
            
    # Ingest sample files if not present in ChromaDB
    for file_name in SAMPLE_DOCS.keys():
        if not store.check_document_exists(file_name):
            file_path = os.path.join(docs_dir, file_name)
            pages = extract_text_from_file(file_path)
            chunks = create_chunks(pages)
            store.add_documents(chunks)
            
    return store

def run_llm_grading(pipeline: RetrievalPipeline, question: str, ground_truth: str, generated_answer: str) -> Tuple[int, str]:
    """
    Asks the LLM to grade the correctness of the generated answer compared to the ground truth.
    Returns: (score [1-5], reasoning)
    """
    prompt = (
        f"You are an expert grading system for RAG systems.\n"
        f"Your task is to grade the correctness of a generated answer compared to a ground truth answer.\n\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Generated Answer: {generated_answer}\n\n"
        f"Grade the correctness on a scale of 1 to 5:\n"
        f"1: The generated answer is completely incorrect, hallucinated, or states it cannot answer when the information was available in the ground truth.\n"
        f"2: The generated answer has minor correct parts but is mostly incorrect or missing critical facts.\n"
        f"3: The generated answer is partially correct (about 50%) but misses key details or includes some incorrect facts.\n"
        f"4: The generated answer is mostly correct and matches the ground truth, with only minor differences or omissions.\n"
        f"5: The generated answer is completely correct, factual, and fully covers the ground truth information.\n\n"
        f"Respond ONLY in JSON format:\n"
        f"{{\n"
        f"  \"score\": 5,\n"
        f"  \"reasoning\": \"A brief explanation of the score.\"\n"
        f"}}\n"
        f"Do not include any markup blocks, markdown formatting, or notes outside the JSON object."
    )
    
    try:
        response_text = pipeline._call_llm(prompt, json_mode=True)
        data = json.loads(response_text)
        score = int(data.get("score", 1))
        reasoning = data.get("reasoning", "")
        return max(1, min(5, score)), reasoning
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return 1, f"Failed to run LLM grading: {e}"

def main():
    logger.info("Setting up evaluation environment...")
    store = setup_evaluation_environment()
    
    # Initialize pipeline
    reranking_enabled = os.environ.get("RERANKING_ENABLED", "true").lower() == "true"
    reranker = None
    if reranking_enabled:
        try:
            reranker = Reranker()
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            
    pipeline = RetrievalPipeline(chroma_store=store, reranker=reranker)
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    logger.info(f"Loaded {len(qa_pairs)} evaluation Q&A pairs.")
    
    results = []
    total_recall = 0
    total_precision = 0
    total_correctness = 0
    
    for idx, item in enumerate(qa_pairs):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_doc = item["expected_document"]
        
        logger.info(f"Evaluating Query {idx + 1}/{len(qa_pairs)}: '{question}'")
        
        # 1. Execute full ask pipeline
        res = pipeline.ask(question)
        generated_answer = res["answer"]
        
        # 2. Get retrieval chunks for metrics calculation
        # Translate query if necessary
        _, eng_question = pipeline.detect_and_translate_query(question)
        retrieved = store.query_collection(eng_question, top_k=10)
        
        if reranker and reranking_enabled:
            try:
                retrieved = reranker.rerank(eng_question, retrieved)
            except Exception:
                pass
                
        # Calculate Recall@10 and Precision@10
        # Recall@10: 1 if the expected document is in retrieved chunks, else 0
        retrieved_files = [c["metadata"]["source_file"] for c in retrieved]
        recall = 1.0 if expected_doc in retrieved_files else 0.0
        
        # Precision@10: Fraction of retrieved chunks belonging to the expected document
        matching_chunks = [f for f in retrieved_files if f == expected_doc]
        precision = len(matching_chunks) / len(retrieved_files) if retrieved_files else 0.0
        
        # 3. LLM Correctness grading
        correctness_score, correctness_reason = run_llm_grading(
            pipeline, question, ground_truth, generated_answer
        )
        # Normalize score (1-5) to 0.0-1.0
        normalized_correctness = (correctness_score - 1.0) / 4.0
        
        total_recall += recall
        total_precision += precision
        total_correctness += normalized_correctness
        
        results.append({
            "id": item["id"],
            "question": question,
            "expected_document": expected_doc,
            "generated_answer": generated_answer,
            "confidence_score": res["confidence_score"],
            "recall@10": recall,
            "precision@10": precision,
            "correctness_score_1_to_5": correctness_score,
            "correctness_normalized": normalized_correctness,
            "correctness_reasoning": correctness_reason
        })
        
    num_queries = len(qa_pairs)
    summary = {
        "average_recall@10": total_recall / num_queries,
        "average_precision@10": total_precision / num_queries,
        "average_correctness": total_correctness / num_queries,
        "total_queries_evaluated": num_queries
    }
    
    report = {
        "summary": summary,
        "results": results
    }
    
    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info("===== EVALUATION SUMMARY =====")
    logger.info(f"Total Queries: {summary['total_queries_evaluated']}")
    logger.info(f"Avg Recall@10: {summary['average_recall@10']:.2%}")
    logger.info(f"Avg Precision@10: {summary['average_precision@10']:.2%}")
    logger.info(f"Avg Correctness Score: {summary['average_correctness']:.2%}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
