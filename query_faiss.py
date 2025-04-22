import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
import ssl
import httpx
import random
import logging
import socket
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pymilvus directly
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

# Get Zilliz connection details from environment variables
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_USER = os.getenv("ZILLIZ_USER")
ZILLIZ_PASSWORD = os.getenv("ZILLIZ_PASSWORD")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("ZILLIZ_COLLECTION")

TEMP_DIR = "temp"
CHUNKS_FILE = os.path.join(TEMP_DIR, "chunks.json")

# Embedding dimension for text-embedding-ada-002
EMBEDDING_DIM = 1536

# Constants for batch processing - reduced batch size for better reliability
BATCH_SIZE = 20  # Process 20 chunks at a time (reduced from 50)
RATE_LIMIT_PAUSE = 2  # Seconds to pause between batches (increased from 1)

# Custom Milvus Retriever class to replace LangChain's implementation
class MilvusRetriever(BaseRetriever):
    """Custom retriever that uses pymilvus directly to retrieve documents."""
    
    def __init__(
        self,
        collection_name: str,
        embedding_function,
        text_field: str = "text",
        vector_field: str = "embedding",
        id_field: str = "id",
        k: int = 8,
        fetch_k: int = 15
    ):
        """Initialize the MilvusRetriever.
        
        Args:
            collection_name: Name of the Milvus collection
            embedding_function: Function to generate embeddings
            text_field: Field name for document text
            vector_field: Field name for vector embeddings
            id_field: Field name for document ID
            k: Number of documents to return
            fetch_k: Number of documents to fetch before reranking
        """
        super().__init__()
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._text_field = text_field
        self._vector_field = vector_field
        self._id_field = id_field
        self._k = k
        self._fetch_k = fetch_k
        self._collection = None
        
        # Connect to Milvus
        self._connect_to_milvus()
        
    @property
    def collection_name(self):
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, value):
        self._collection_name = value
    
    @property
    def embedding_function(self):
        return self._embedding_function
    
    @embedding_function.setter
    def embedding_function(self, value):
        self._embedding_function = value
    
    @property
    def text_field(self):
        return self._text_field
    
    @text_field.setter
    def text_field(self, value):
        self._text_field = value
    
    @property
    def vector_field(self):
        return self._vector_field
    
    @vector_field.setter
    def vector_field(self, value):
        self._vector_field = value
    
    @property
    def id_field(self):
        return self._id_field
    
    @id_field.setter
    def id_field(self, value):
        self._id_field = value
    
    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, value):
        self._k = value
    
    @property
    def fetch_k(self):
        return self._fetch_k
    
    @fetch_k.setter
    def fetch_k(self, value):
        self._fetch_k = value
        
    @property
    def collection(self):
        return self._collection
    
    @collection.setter
    def collection(self, value):
        self._collection = value
    
    def _connect_to_milvus(self):
        """Establish connection to Milvus server."""
        connections.connect(
            alias="default",
            uri=ZILLIZ_URI,
            user=ZILLIZ_USER,
            password=ZILLIZ_PASSWORD,
            token=ZILLIZ_TOKEN,
            secure=True
        )
        
        # Load the collection
        self._collection = Collection(self._collection_name)
        self._collection.load()
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query.
        
        Args:
            query: Query string
            
        Returns:
            List of relevant documents
        """
        # Generate embeddings for the query
        query_embedding = self._embedding_function.embed_query(query)
        
        # Search in Milvus
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        # Execute search
        results = self._collection.search(
            data=[query_embedding],
            anns_field=self._vector_field,
            param=search_params,
            limit=self._fetch_k,
            output_fields=[self._text_field]
        )
        
        # Convert results to documents
        docs = []
        for hits in results:
            for hit in hits:
                text = hit.entity.get(self._text_field)
                if text:
                    docs.append(Document(page_content=text))
        
        # Return top k documents
        return docs[:self._k]

# Define a retry decorator for embedding generation with improved parameters
@retry(
    stop=stop_after_attempt(20),  # Increased retries from 15 to 20
    wait=wait_exponential(multiplier=1, min=4, max=180),  # Extended max backoff to 3 minutes
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=tenacity.retry_if_exception_type((
        Exception, 
        ssl.SSLError, 
        httpx.ReadTimeout, 
        httpx.ConnectTimeout, 
        httpx.ReadError,
        httpx.WriteError,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        socket.timeout,
        socket.error,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        ConnectionAbortedError,
        TimeoutError,  # Added for general timeout errors
        OSError,       # Added for underlying OS-level network errors
        BrokenPipeError  # Added specifically for pipe failures
    ))  # Expanded list of network-related errors
)
def generate_embeddings_with_retry(embeddings, text):
    """Generate embeddings with retry logic for API failures and network issues."""
    try:
        # Set a longer timeout for the embedding request
        if hasattr(embeddings, 'client'):
            old_timeout = embeddings.client.timeout if hasattr(embeddings.client, 'timeout') else None
            if hasattr(embeddings.client, 'timeout'):
                embeddings.client.timeout = 120.0  # Increased to 120-second timeout
            
            # If client has session attribute, attempt to close and recreate it
            if hasattr(embeddings.client, 'session') and hasattr(embeddings.client.session, 'close'):
                try:
                    embeddings.client.session.close()
                    # Force recreation of session on next use
                    embeddings.client._session = None
                except Exception as session_err:
                    logger.warning(f"Failed to reset session: {str(session_err)}")
        
        # Attempt to generate embeddings
        result = embeddings.embed_query(text)
        
        # Reset timeout to original value if we changed it
        if hasattr(embeddings, 'client') and old_timeout is not None and hasattr(embeddings.client, 'timeout'):
            embeddings.client.timeout = old_timeout
            
        return result
        
    except Exception as e:
        logger.warning(f"Embedding retry triggered: {str(e)}")
        logger.warning(f"Error type: {type(e).__name__}")
        
        # Add a longer sleep before retry to help with possible rate limiting or network issues
        sleep_time = random.uniform(10, 30)  # Increased random sleep between 10-30 seconds
        logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry...")
        time.sleep(sleep_time)
        
        # Clear any potential SSL session cache if it's an SSL error
        if isinstance(e, ssl.SSLError):
            try:
                logger.info("Clearing SSL session cache before retry...")
                ssl.SSLContext.session_stats({})  # This will help reset SSL sessions
            except Exception as ssl_err:
                logger.warning(f"Failed to clear SSL session: {str(ssl_err)}")
        
        # Make sure we re-raise so the retry decorator can do its job
        raise

def create_faiss_index(max_chunks=None):
    """Loads text chunks, generates embeddings, and stores them in Zilliz Milvus using pymilvus directly.
    Includes batch processing to handle large books more efficiently.
    
    Args:
        max_chunks: Optional maximum number of chunks to process (for very large books)
    """
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    # Load stored chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Chunks file is empty. Cannot create vector index.")
        
    # Apply chunk limit if specified
    if max_chunks and max_chunks > 0 and len(chunks) > max_chunks:
        print(f"Limiting processing to first {max_chunks} chunks (out of {len(chunks)} total)")
        chunks = chunks[:max_chunks]

    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-ada-002", timeout=60)
    
    # Generate vector embeddings for each chunk in batches
    embedded_chunks = []
    total_chunks = len(chunks)
    
    print(f"Processing {total_chunks} chunks in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_chunks+BATCH_SIZE-1)//BATCH_SIZE}: chunks {i} to {min(i+BATCH_SIZE-1, total_chunks-1)}")
        
        batch_embedded_chunks = []
        retry_count = 0
        
        # Keep track of successful and failed chunks
        successful_chunks = 0
        failed_chunks = 0
        
        for j, chunk in enumerate(batch):
            try:
                # Use retry-enabled function
                embedding = generate_embeddings_with_retry(embeddings, chunk)
                batch_embedded_chunks.append({
                    "id": i + j,
                    "text": chunk,
                    "embedding": embedding
                })
                successful_chunks += 1
                print(f"  Processed chunk {i + j + 1}/{total_chunks}")
            except Exception as e:
                failed_chunks += 1
                print(f"Error embedding chunk {i + j} after multiple retries: {str(e)}")
                # Add the chunk with a placeholder embedding to avoid data loss
                # This will be less effective but prevents total failure
                batch_embedded_chunks.append({
                    "id": i + j,
                    "text": chunk,
                    "embedding": [0.0] * EMBEDDING_DIM
                })
        
        embedded_chunks.extend(batch_embedded_chunks)
        print(f"Batch completed: {successful_chunks} successful, {failed_chunks} failed")
        
        # Sleep between batches to respect rate limits
        if i + BATCH_SIZE < total_chunks:
            sleep_time = RATE_LIMIT_PAUSE + (failed_chunks * 0.5)  # Adaptive wait time
            print(f"Pausing for {sleep_time} seconds to respect rate limits...")
            time.sleep(sleep_time)

    # Connect to Milvus
    try:
        connections.connect(
            alias="default",
            uri=ZILLIZ_URI,
            user=ZILLIZ_USER,
            password=ZILLIZ_PASSWORD,
            token=ZILLIZ_TOKEN,
            secure=True
        )
        
        # Check if collection exists and drop if needed
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
        
        # Define collection schema
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema = CollectionSchema(fields=[id_field, text_field, embedding_field], description="Text chunks with embeddings")
        
        # Create collection
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Create index on vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        # Insert data into collection in batches with error handling
        milvus_batch_size = 50  # Smaller batch size for insertion
        for i in range(0, len(embedded_chunks), milvus_batch_size):
            try:
                batch = embedded_chunks[i:i + milvus_batch_size]
                batch_data = [
                    [item["id"] for item in batch],
                    [item["text"] for item in batch],
                    [item["embedding"] for item in batch]
                ]
                
                collection.insert(batch_data)
                print(f"Inserted batch {i//milvus_batch_size + 1}/{(len(embedded_chunks)+milvus_batch_size-1)//milvus_batch_size}")
            except Exception as e:
                print(f"Error inserting batch {i//milvus_batch_size + 1}: {str(e)}")
                # Continue with next batch rather than failing completely
        
        collection.flush()
        
        print(f"Vectors stored in Zilliz collection: {COLLECTION_NAME}")
        return COLLECTION_NAME
    except Exception as e:
        print(f"Error storing vectors in Milvus: {str(e)}")
        raise

def get_answer(question):
    """Retrieves the best answer from Zilliz Milvus for a given question using an LLM."""
    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY, model="text-embedding-ada-002", timeout=60)
    
    # Connect to Milvus collection
    retriever = MilvusRetriever(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        text_field="text",
        k=8,
        fetch_k=15
    )
    
    try:
        # Use our retry function for the query embedding
        query_embedding = generate_embeddings_with_retry(embeddings, question)
        retrieved_docs = retriever.get_relevant_documents(question)
        print("Retrieved Chunks:", [doc.page_content[:100] + "..." for doc in retrieved_docs])  # Truncate for logging
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        # Fall back to default response if retrieval fails
        return "I'm sorry, I encountered an issue retrieving information from the document. Please try again in a moment."

    # Initialize LLM with increased timeout
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        openai_api_key=OPENAI_KEY, 
        temperature=0.3,
        request_timeout=90  # Increase timeout for LLM calls
    )

    # Improved system prompt that doesn't emphasize lack of information
    system_prompt = (
    "You are a knowledgeable assistant that provides insightful answers about books.\n"
    "Use the provided book content to answer questions confidently and thoughtfully.\n"
    "When summarizing, extract key points, themes, and important details from the available content.\n"
    "Connect related information from different parts of the text to provide comprehensive answers.\n"
    "Focus on what you can determine from the text rather than emphasizing what might be missing.\n"
    "If certain details aren't available, simply focus on what IS available without disclaimers.\n"
    "Make reasonable inferences based on the text, but avoid inventing specific plot points or quotes.\n\n"
    "Book Content:\n{context}\n\n"
    )

    # Create a Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    try:
        # Create a chain that retrieves documents and processes them with the LLM
        qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        
        # Get AI-generated answer with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = qa_chain.invoke({"input": question})["answer"]
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"LLM query failed, retrying ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(5 * (attempt + 1))  # Progressive backoff
                else:
                    raise
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "I apologize, but I'm having trouble generating a response at the moment. Please try again shortly."
