from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
import io

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface

router = APIRouter(prefix="/api/v1", tags=["services"])


# ============================================================================
# Embedding Endpoints
# ============================================================================

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to generate embedding for")


@router.post("/embedding")
async def get_embedding(request: EmbeddingRequest):
    """Generate embedding for a single text."""
    try:
        result = await OrchestratorInterface.get_embedding(request.text)
        return {"embedding": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EmbeddingsRequest(BaseModel):
    text_list: List[str] = Field(..., description="List of texts to generate embeddings for")


@router.post("/embeddings")
async def get_embeddings(request: EmbeddingsRequest):
    """Generate embeddings for multiple texts."""
    try:
        result = await OrchestratorInterface.get_embeddings(request.text_list)
        return {"embeddings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Reranking Endpoints
# ============================================================================

class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text for reranking")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_k: int = Field(..., description="Number of top documents to return")


@router.post("/rerank")
async def rerank(request: RerankRequest):
    """Rerank documents based on query relevance."""
    try:
        result = await OrchestratorInterface.rerank(
            request.query,
            request.documents,
            request.top_k
        )
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# CLIP Endpoints
# ============================================================================

@router.post("/clip/score")
async def get_clip_score(
    text: str = Form(..., description="Text to compare with image"),
    image: UploadFile = File(..., description="Image file")
):
    """Calculate CLIP similarity score between an image and text."""
    try:
        # Load image from upload
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        result = await OrchestratorInterface.get_clip_score(img, text)
        return {"score": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clip/scores")
async def get_clip_scores(
    texts: str = Form(..., description="Comma-separated list of texts to compare with image"),
    image: UploadFile = File(..., description="Image file")
):
    """Calculate CLIP similarity scores between an image and multiple texts."""
    try:
        # Load image from upload
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Parse comma-separated texts
        text_list = [text.strip() for text in texts.split(',')]

        result = await OrchestratorInterface.get_clip_scores(img, text_list)
        return {"scores": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Clustering Algorithm Endpoints
# ============================================================================

class KMeansRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    n_clusters: int = Field(..., description="Number of clusters")


@router.post("/algorithm/k-means")
async def k_means_clustering(request: KMeansRequest):
    """Perform K-Means clustering on data."""
    try:
        result = await OrchestratorInterface.k_means(request.data, request.n_clusters)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgglomerativeRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    n_clusters: int = Field(..., description="Number of clusters")
    linkage: str = Field(default="ward", description="Linkage method: ward, complete, average, single")


@router.post("/algorithm/agglomerative")
async def agglomerative_clustering(request: AgglomerativeRequest):
    """Perform Agglomerative clustering on data."""
    try:
        result = await OrchestratorInterface.agglomerative(
            request.data,
            request.n_clusters,
            request.linkage
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AutoAgglomerativeRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    max_clusters: int = Field(..., description="Maximum number of clusters to evaluate")
    linkage: str = Field(default="ward", description="Linkage method: ward, complete, average, single")


@router.post("/algorithm/auto-agglomerative")
async def auto_agglomerative_clustering(request: AutoAgglomerativeRequest):
    """Perform Agglomerative clustering with automatic cluster number selection."""
    try:
        result = await OrchestratorInterface.auto_agglomerative(
            request.data,
            request.max_clusters,
            request.linkage
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Bucketing Algorithm Endpoints
# ============================================================================

class SimilarityBucketingRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    buckets: List[dict] = Field(..., description="Bucket definitions with 'bucket_name' and 'bucket_embedding' fields")
    score_method: str = Field(default="cosine", description="Scoring method: cosine or dot_product")
    allow_orphan_bucket: bool = Field(default=False, description="Allow orphan bucket for low-similarity items")
    orphan_threshold: float = Field(default=0.5, description="Threshold for orphan bucket assignment")


@router.post("/algorithm/similarity-bucketing")
async def similarity_bucketing(request: SimilarityBucketingRequest):
    """Perform similarity-based bucketing on data."""
    try:
        result = await OrchestratorInterface.similarity_bucketing(
            request.data,
            request.buckets,
            request.score_method,
            request.allow_orphan_bucket,
            request.orphan_threshold
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
