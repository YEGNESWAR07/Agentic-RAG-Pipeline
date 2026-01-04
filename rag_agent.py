"""
Agentic RAG Pipeline - Local, Privacy-Friendly Document Q&A System

This module provides an intelligent document retrieval and question-answering system
using LangChain, ChromaDB, and local language models.
"""
import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline

# Configuration constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"
DEFAULT_RETRIEVAL_K = 3
DEFAULT_MAX_TOKENS = 150

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a query operation with metadata."""
    query: str
    answer: str
    action: str  # 'search' or 'direct'
    confidence: float
    sources_used: int
    retrieval_time: float = 0.0
    generation_time: float = 0.0


def load_docs(folder_path: str) -> List[Document]:
    """
    Load all PDF documents from a folder.
    
    Args:
        folder_path: Path to folder containing PDF files
        
    Returns:
        List of loaded documents
        
    Raises:
        FileNotFoundError: If folder doesn't exist
        ValueError: If no PDF files found
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    
    for file in pdf_files:
        path = os.path.join(folder_path, file)
        try:
            logger.info(f"Loading: {file}")
            loader = PyPDFLoader(path)
            file_docs = loader.load()
            docs.extend(file_docs)
            logger.info(f"  ✓ Loaded {len(file_docs)} pages from {file}")
        except Exception as e:
            logger.error(f"  ✗ Error loading {file}: {e}")
            continue
    
    if not docs:
        raise ValueError("No documents were successfully loaded")
    
    return docs


def create_chunks(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of document chunks
    """
    if not docs:
        raise ValueError("No documents provided for chunking")
    
    logger.info(f"Creating chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    
    return chunks


def build_vectorstore(
    chunks: List[Document],
    persist_directory: str = "./chroma_db",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> Chroma:
    """
    Build and persist a vector store from document chunks.
    
    Args:
        chunks: List of document chunks
        persist_directory: Directory to persist the vector store
        embedding_model: Name of the embedding model to use
        
    Returns:
        Chroma retriever instance
    """
    if not chunks:
        raise ValueError("No chunks provided for vectorstore creation")
    
    logger.info(f"Building vector store with {len(chunks)} chunks")
    logger.info(f"Using embedding model: {embedding_model}")
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create vectorstore
        db = Chroma(
            collection_name="rag_store",
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            db.add_texts(batch_texts, metadatas=batch_metadatas)
            logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        logger.info(f"✓ Vector store built and persisted to {persist_directory}")
        return db
        
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        raise


def load_vectorstore(
    persist_directory: str = "./chroma_db",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> Optional[Chroma]:
    """
    Load an existing vector store from disk.
    
    Args:
        persist_directory: Directory where vector store is persisted
        embedding_model: Name of the embedding model to use
        
    Returns:
        Chroma retriever instance or None if not found
    """
    if not os.path.exists(persist_directory):
        logger.warning(f"Vector store not found at {persist_directory}")
        return None
    
    try:
        logger.info(f"Loading vector store from {persist_directory}")
        embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        
        db = Chroma(
            collection_name="rag_store",
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
        
        logger.info("✓ Vector store loaded successfully")
        return db
        
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def agent_controller(query: str) -> str:
    """
    Determine whether to search documents or answer directly.
    
    Args:
        query: User's question
        
    Returns:
        'search' or 'direct'
    """
    q = query.lower().strip()
    
    # First, check for patterns that clearly indicate general knowledge questions
    # These should be answered directly without searching documents
    general_knowledge_patterns = [
        # Math questions
        r'\d+\s*[\+\-\*\/]\s*\d+',  # "2+2", "10 * 5", etc.
        r'what is \d+',  # "what is 5", etc.
        r'calculate',
        
        # General facts
        r'who is',
        r'who was',
        r'when did',
        r'where is',
        r'capital of',
        r'define ',
        r'meaning of',
        
        # Greetings and simple queries
        r'^(hi|hello|hey|thanks|thank you)',
    ]
    
    import re
    for pattern in general_knowledge_patterns:
        if re.search(pattern, q):
            return "direct"
    
    # Keywords that strongly suggest document search is needed
    document_specific_keywords = [
        "pdf", "document", "file",
        "summarize", "summary", 
        "according to the", "based on the", "from the",
        "in the document", "in the pdf", "in the manual",
        "invoice", "report", "policy", "manual", "catalog",
        "meeting", "financial", "training", "faq"
    ]
    
    # Check for document-specific keywords
    if any(keyword in q for keyword in document_specific_keywords):
        return "search"
    
    # Phrases that suggest looking for specific information in documents
    document_phrases = [
        "what is the total",
        "what is the amount",
        "what are the action items",
        "what are the key findings",
        "list the",
        "extract",
        "find information about",
        "tell me about the",
        "show me the",
        "what does the",
        "how does the"
    ]
    
    if any(phrase in q for phrase in document_phrases):
        return "search"
    
    # Default to direct answer for general questions
    return "direct"


def handle_arithmetic(query: str) -> Optional[str]:
    """
    Detect and solve simple arithmetic questions.
    
    Args:
        query: User's question
        
    Returns:
        Answer string if arithmetic detected, None otherwise
    """
    import re
    
    q = query.lower().strip()
    
    # Pattern to match simple arithmetic: "what is X+Y", "X+Y", "calculate X+Y", etc.
    patterns = [
        r'what\s+is\s+(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)',
        r'calculate\s+(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)',
        r'^(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)$',
        r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)\s*[\?=]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            try:
                num1 = float(match.group(1))
                operator = match.group(2)
                num2 = float(match.group(3))
                
                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator == '*':
                    result = num1 * num2
                elif operator == '/':
                    if num2 == 0:
                        return "Cannot divide by zero."
                    result = num1 / num2
                else:
                    return None
                
                # Format result nicely (remove .0 for whole numbers)
                if result == int(result):
                    result = int(result)
                
                return f"The answer is {result}."
                
            except (ValueError, ZeroDivisionError):
                return None
    
    return None



def calculate_confidence(
    query: str,
    answer: str,
    retrieved_docs: List[Document],
    action: str
) -> float:
    """
    Calculate confidence score for the answer.
    
    Args:
        query: Original query
        answer: Generated answer
        retrieved_docs: Documents retrieved (if any)
        action: Action taken ('search' or 'direct')
        
    Returns:
        Confidence score between 0 and 1
    """
    confidence = 0.5  # Base confidence
    
    # Higher confidence for search-based answers with multiple sources
    if action == "search" and retrieved_docs:
        # More sources = higher confidence
        source_bonus = min(len(retrieved_docs) * 0.1, 0.3)
        confidence += source_bonus
        
        # Check if answer contains specific information (numbers, dates, names)
        if any(char.isdigit() for char in answer):
            confidence += 0.1
    
    # Lower confidence for very short answers
    if len(answer.split()) < 10:
        confidence -= 0.1
    
    # Higher confidence for longer, detailed answers
    if len(answer.split()) > 30:
        confidence += 0.1
    
    # Ensure confidence is between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    return round(confidence, 2)


def rag_answer(
    query: str,
    llm,
    retriever: Optional[Chroma] = None,
    k: int = DEFAULT_RETRIEVAL_K
) -> QueryResult:
    """
    Generate an answer using the RAG pipeline with agentic decision-making.
    
    Args:
        query: User's question
        llm: Language model pipeline
        retriever: Vector store retriever (optional)
        k: Number of documents to retrieve
        
    Returns:
        QueryResult with answer and metadata
    """
    import time
    
    # First, check if this is a simple arithmetic question we can solve directly
    arithmetic_answer = handle_arithmetic(query)
    if arithmetic_answer:
        logger.info(f"🧮 Detected arithmetic question, computing directly: '{query}'")
        return QueryResult(
            query=query,
            answer=arithmetic_answer,
            action="direct",
            confidence=1.0,  # 100% confidence for computed answers
            sources_used=0,
            retrieval_time=0.0,
            generation_time=0.0
        )
    
    action = agent_controller(query)
    retrieved_docs = []
    retrieval_time = 0.0
    
    if action == "search":
        logger.info(f"🕵️  Agent decided to SEARCH documents for: '{query}'")
        
        if retriever is None:
            logger.warning("⚠️  No retriever available, falling back to direct answer")
            action = "direct"
        else:
            try:
                start_time = time.time()
                retriever_obj = retriever.as_retriever(search_kwargs={"k": k})
                retrieved_docs = retriever_obj.get_relevant_documents(query)
                retrieval_time = time.time() - start_time
                
                logger.info(f"  Retrieved {len(retrieved_docs)} relevant chunks")
                
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                final_prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
            except Exception as e:
                logger.error(f"Error during retrieval: {e}")
                action = "direct"
                final_prompt = query
    else:
        logger.info(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query
    
    # Generate answer
    try:
        gen_start = time.time()
        response = llm(final_prompt, max_new_tokens=DEFAULT_MAX_TOKENS)[0]["generated_text"]
        generation_time = time.time() - gen_start
        
        # Clean up response (remove prompt if model echoes it)
        if final_prompt in response:
            response = response.replace(final_prompt, "").strip()
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        response = f"Error generating answer: {e}"
        generation_time = 0.0
    
    # Calculate confidence
    confidence = calculate_confidence(query, response, retrieved_docs, action)
    
    return QueryResult(
        query=query,
        answer=response,
        action=action,
        confidence=confidence,
        sources_used=len(retrieved_docs),
        retrieval_time=retrieval_time,
        generation_time=generation_time
    )


def print_result(result: QueryResult, verbose: bool = True):
    """
    Print query result in a formatted way.
    
    Args:
        result: QueryResult to print
        verbose: Whether to print detailed metadata
    """
    print("\n" + "=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(result.answer)
    print("=" * 70)
    
    if verbose:
        print(f"\nMetadata:")
        print(f"  Action: {result.action.upper()}")
        print(f"  Confidence: {result.confidence * 100:.1f}%")
        print(f"  Sources Used: {result.sources_used}")
        if result.retrieval_time > 0:
            print(f"  Retrieval Time: {result.retrieval_time:.2f}s")
        print(f"  Generation Time: {result.generation_time:.2f}s")
        print(f"  Total Time: {result.retrieval_time + result.generation_time:.2f}s")
    print()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG Pipeline - Local, Privacy-Friendly Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from PDFs
  python rag_agent.py build --data-dir ./data

  # Query with a specific question
  python rag_agent.py query --q "What is the total amount in the invoice?"

  # Interactive mode
  python rag_agent.py query
        """
    )
    
    sub = parser.add_subparsers(dest="cmd", help="Command to execute")

    # Build command
    build = sub.add_parser("build", help="Build vector store from PDFs")
    build.add_argument(
        "--data-dir",
        default="./data",
        help="Folder containing PDF files (default: ./data)"
    )
    build.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Chroma DB persist directory (default: ./chroma_db)"
    )
    build.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for splitting documents (default: {DEFAULT_CHUNK_SIZE})"
    )
    build.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})"
    )

    # Query command
    query = sub.add_parser("query", help="Query the agent")
    query.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Chroma DB persist directory (default: ./chroma_db)"
    )
    query.add_argument(
        "--q",
        required=False,
        help="One-off query (if omitted, starts interactive mode)"
    )
    query.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_RETRIEVAL_K})"
    )
    query.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metadata for each answer"
    )

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    # Build command
    if args.cmd == "build":
        try:
            logger.info("=" * 70)
            logger.info("BUILDING VECTOR STORE")
            logger.info("=" * 70)
            
            docs = load_docs(args.data_dir)
            logger.info(f"✓ Loaded {len(docs)} pages from PDFs")
            
            chunks = create_chunks(docs, args.chunk_size, args.chunk_overlap)
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            build_vectorstore(chunks, persist_directory=args.persist_dir)
            
            logger.info("=" * 70)
            logger.info("✓ BUILD COMPLETE")
            logger.info("=" * 70)
            logger.info(f"\nYou can now query using:")
            logger.info(f'  python rag_agent.py query --q "Your question here"')
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            sys.exit(1)

    # Query command
    elif args.cmd == "query":
        try:
            # Load LLM
            logger.info("Loading language model (this may take a moment)...")
            
            # Try to use GPU if available, otherwise fall back to CPU
            try:
                llm = pipeline(
                    "text2text-generation",
                    model=DEFAULT_LLM_MODEL,
                    device_map="auto"  # Automatically use GPU if available
                )
            except Exception:
                # Fall back to CPU if device_map fails
                logger.info("GPU not available or accelerate not installed, using CPU...")
                llm = pipeline(
                    "text2text-generation",
                    model=DEFAULT_LLM_MODEL
                )
            
            logger.info("✓ Language model loaded")
            
            # Load vector store
            retriever = load_vectorstore(persist_directory=args.persist_dir)
            
            if retriever is None:
                logger.warning("⚠️  No vector store found. Agent will answer directly without document search.")
                logger.warning(f"   Build a vector store first using: python rag_agent.py build")
            
            # One-off query
            if args.q:
                result = rag_answer(args.q, llm, retriever, k=args.k)
                print_result(result, verbose=args.verbose)
                return
            
            # Interactive mode
            logger.info("\n" + "=" * 70)
            logger.info("INTERACTIVE MODE")
            logger.info("=" * 70)
            logger.info("Type your questions below. Type 'exit' or 'quit' to stop.\n")
            
            while True:
                try:
                    q = input("You: ").strip()
                    
                    if not q:
                        continue
                    
                    if q.lower() in ("exit", "quit", "q"):
                        logger.info("Goodbye!")
                        break
                    
                    result = rag_answer(q, llm, retriever, k=args.k)
                    print_result(result, verbose=args.verbose)
                    
                except KeyboardInterrupt:
                    logger.info("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Query failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
