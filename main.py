import asyncio
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from frontend import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=1000, description="Search query text"
    )
    num_results: int = Field(
        default=10, ge=1, le=100, description="Number of results to return"
    )

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class AddDocsRequest(BaseModel):
    docs: List[str] = Field(
        ..., min_items=1, max_items=1000, description="List of documents to add"
    )

    @validator("docs")
    def validate_docs(cls, v):
        # Filter out empty documents
        filtered_docs = [doc.strip() for doc in v if doc and doc.strip()]
        if not filtered_docs:
            raise ValueError("At least one non-empty document is required")
        return filtered_docs


class DocumentResult(BaseModel):
    id: int
    doc_id: int
    location: str
    text: str
    section: str
    metadata: Dict[str, Any] = {}
    timestamp: str
    score: Optional[float] = None  # Similarity score from retrieval


class SearchResponse(BaseModel):
    results: List[DocumentResult]
    query: str
    total_results: int
    search_time_ms: float
    timestamp: str


class AddDocsResponse(BaseModel):
    success: bool
    docs_added: int
    message: str
    processing_time_ms: float


class StatsResponse(BaseModel):
    total_documents: int
    database_size_mb: Optional[float] = None
    last_updated: Optional[str] = None
    schema_info: Dict[str, str]


# Initialize FastAPI app
app = FastAPI(
    title="Local RAG API",
    description="FastAPI backend for local document search",
    version="0.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


class PyArrowVectorDBWrapper:
    """
    Wrapper class for your existing PyArrow vector database client
    """

    def __init__(self, vector_db_client):
        """
        Initialize with your existing vector DB client

        Args:
            vector_db_client: Your existing PyArrow vector DB client with add_docs() and retrieve() methods
        """
        self.client = vector_db_client
        self.is_ready = True
        logger.info("PyArrow Vector DB wrapper initialized")

    async def add_documents_async(self, docs: List[List[bytes | str]]) -> Dict[str, Any]:
        """
        Async wrapper for add_docs method
        """
        start_time = time.time()

        try:
            # Run the blocking add_docs operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, self.client.add_docs, docs)

            processing_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "docs_added": len(docs),
                "processing_time_ms": processing_time,
                "result": result,
            }

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error adding documents: {e}\n\n{traceback.format_exc()}")
            return {
                "success": False,
                "docs_added": 0,
                "processing_time_ms": processing_time,
                "error": str(e),
            }

    async def search_documents_async(
        self, query: str, n_docs: int
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper for retrieve method
        """
        try:
            # Run the blocking retrieve operation in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor, self.client.retrieve, query, n_docs
            )

            # Convert results to expected format
            formatted_results = []
            for result in results:
                # Parse metadata if it's a JSON string
                metadata = {}
                if isinstance(result.get("metadata"), str):
                    try:
                        metadata = json.loads(result["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        metadata = {"raw": result.get("metadata", "")}
                elif isinstance(result.get("metadata"), dict):
                    metadata = result["metadata"]

                # Format timestamp
                timestamp_str = result.get("timestamp", datetime.now()).isoformat()
                if hasattr(result.get("timestamp"), "isoformat"):
                    timestamp_str = result["timestamp"].isoformat()

                formatted_result = {
                    "id": result.get("id", 0),
                    "doc_id": result.get("doc_id", 0),
                    "location": result.get("location", ""),
                    "text": result.get("text", ""),
                    "section": result.get("section", ""),
                    "metadata": metadata,
                    "timestamp": timestamp_str,
                    "score": result.get("score"),  # Include similarity score if available
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}\n\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        """
        try:
            # Try to get stats from your client if it has such methods
            stats = {
                "total_documents": getattr(self.client, "document_count", 0),
                "schema_info": {
                    "id": "int64",
                    "doc_id": "int64",
                    "location": "string",
                    "text": "string",
                    "embedding": "list[float32]",
                    "section": "string",
                    "metadata": "string (JSON)",
                    "timestamp": "timestamp(us)",
                },
            }

            # Add more stats if your client supports them
            if hasattr(self.client, "get_database_size"):
                stats["database_size_mb"] = self.client.get_database_size()

            if hasattr(self.client, "last_updated"):
                stats["last_updated"] = self.client.last_updated

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_documents": 0, "schema_info": {}, "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Initialize any startup tasks"""

    if vector_db is None:
        logger.error("Vector database not initialized!")
    else:
        logger.info("FastAPI server started with PyArrow Vector DB")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PyArrow Vector DB API is running",
        "status": "healthy" if vector_db and vector_db.is_ready else "error",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not vector_db:
        return {
            "status": "error",
            "message": "Vector database not initialized",
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "status": "healthy" if vector_db.is_ready else "error",
        "database_ready": vector_db.is_ready,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/query", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for similar documents using semantic similarity
    """
    if not vector_db or not vector_db.is_ready:
        raise HTTPException(status_code=503, detail="Vector database not available")

    start_time = time.time()
    logger.info(
        f"Searching for: '{request.num_results}' entries for query '{request.query}'"
    )

    try:
        # Perform the search
        results = await vector_db.search_documents_async(
            request.query, request.num_results
        )

        # Calculate response time
        search_time = (time.time() - start_time) * 1000

        # Convert to response model
        document_results = [DocumentResult(**result) for result in results]

        logger.info(
            f"Search completed in {search_time:.2f}ms, found {len(results)} results"
        )

        return SearchResponse(
            results=document_results,
            query=request.query,
            total_results=len(results),
            search_time_ms=search_time,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        search_time = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# TODO: add URL upload
@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Upload documents from a bunch of files
    """
    if not vector_db or not vector_db.is_ready:
        raise HTTPException(status_code=503, detail="Vector database not available")

    try:
        # Read file content
        # TODO: add support for multi file upload
        content = [[await file.read(), file.filename] for file in files]

        # Add documents in background
        background_tasks.add_task(vector_db.add_documents_async, content)

        return {
            "message": f"Upload initiated for {len(files)} documents",
            "docs_count": len(files),
            "status": "processing",
        }

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_database_stats():
    """Get database statistics"""
    if not vector_db:
        raise HTTPException(status_code=503, detail="Vector database not available")

    try:
        stats = vector_db.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


try:
    vector_client = Client()
    vector_db = PyArrowVectorDBWrapper(vector_client)
except Exception as e:
    logger.error(f"Failed to initialize vector database: {e}")
    vector_db = None

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8081

    # Run the server
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # Enable auto-reload during development
        workers=1,  # Single worker for development
        log_level="info",
    )
