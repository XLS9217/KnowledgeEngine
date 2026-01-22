from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from PIL import Image
import io

from src.task_orchestrator.orchestrator_interface import OrchestratorInterface

router = APIRouter(tags=["services"])


# ============================================================================
# Embedding Endpoints
# ============================================================================

class EmbeddingRequest(BaseModel):
    text: str = Field(..., description="Text to generate embedding for")


class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(..., description="Generated embedding vector")
    model_name: Optional[str] = Field(None, description="Name/ID of the model used")


@router.post("/embedding", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embedding for a single text."""
    try:
        result = await OrchestratorInterface.get_embedding(request.text)
        # result may be a dict with keys: model_name, result; or a raw embedding list
        if isinstance(result, dict):
            # Prefer model_name from orchestrator; fallback to engine's currently loaded model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.embedding_model, "model_name", None)
            except Exception:
                engine_model_name = None
            model_name = result.get("model_name") or engine_model_name
            return EmbeddingResponse(embedding=result.get("result"), model_name=model_name)
        else:
            # Fallback: obtain model_name from the currently loaded engine model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.embedding_model, "model_name", None)
            except Exception:
                engine_model_name = None
            return EmbeddingResponse(embedding=result, model_name=engine_model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EmbeddingsRequest(BaseModel):
    text_list: List[str] = Field(..., description="List of texts to generate embeddings for")


class EmbeddingItem(BaseModel):
    text: str = Field(..., description="Original text")
    embedding: List[float] = Field(..., description="Embedding vector for the text")


class EmbeddingsResponse(BaseModel):
    embeddings: List[EmbeddingItem] = Field(..., description="List of text-embedding pairs")
    model_name: Optional[str] = Field(None, description="Name/ID of the model used")


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def get_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Generate embeddings for multiple texts."""
    try:
        result = await OrchestratorInterface.get_embeddings(request.text_list)
        # result is expected to be a dict with keys: model_name, result (list of {text, embedding})
        items = result.get("result", []) if isinstance(result, dict) else result
        if isinstance(result, dict):
            model_name = result.get("model_name")
        else:
            # Fallback: obtain model_name from the currently loaded engine model
            try:
                model_name = getattr(OrchestratorInterface.engine.embedding_model, "model_name", None)
            except Exception:
                model_name = None
        embeddings = [EmbeddingItem(text=item["text"], embedding=item["embedding"]) for item in items]
        return EmbeddingsResponse(embeddings=embeddings, model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Reranking Endpoints
# ============================================================================

class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text for reranking")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_k: int = Field(..., description="Number of top documents to return")


class RerankResponse(BaseModel):
    results: List[Tuple[int, float, str]] = Field(..., description="Ranked results as (index, score, document) tuples")
    model_name: Optional[str] = Field(None, description="Name/ID of the model used")


@router.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    """Rerank documents based on query relevance."""
    try:
        result = await OrchestratorInterface.rerank(
            request.query,
            request.documents,
            request.top_k
        )
        # result is expected to be a dict with keys: model_name, result (list of tuples)
        if isinstance(result, dict):
            return RerankResponse(results=result.get("result"), model_name=result.get("model_name"))
        else:
            # Fallback: obtain model_name from the currently loaded engine model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.reranker_model, "model_name", None)
            except Exception:
                engine_model_name = None
            return RerankResponse(results=result, model_name=engine_model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# CLIP Endpoints
# ============================================================================

class ClipScoreResponse(BaseModel):
    score: float = Field(..., description="CLIP similarity score")
    model_name: Optional[str] = Field(None, description="Name/ID of the model used")


class ClipScoresResponse(BaseModel):
    scores: List[Tuple[str, float]] = Field(..., description="List of (text, score) pairs")
    model_name: Optional[str] = Field(None, description="Name/ID of the model used")


@router.post("/clip/score", response_model=ClipScoreResponse)
async def get_clip_score(
    text: str = Form(..., description="Text to compare with image"),
    image: UploadFile = File(..., description="Image file")
) -> ClipScoreResponse:
    """Calculate CLIP similarity score between an image and text."""
    try:
        # Load image from upload
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        result = await OrchestratorInterface.get_clip_score(img, text)
        if isinstance(result, dict):
            # Prefer model_name from orchestrator; fallback to engine's currently loaded model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.clip_model, "model_name", None)
            except Exception:
                engine_model_name = None
            model_name = result.get("model_name") or engine_model_name
            return ClipScoreResponse(score=result.get("result"), model_name=model_name)
        else:
            # Fallback: obtain model_name from the currently loaded engine model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.clip_model, "model_name", None)
            except Exception:
                engine_model_name = None
            return ClipScoreResponse(score=result, model_name=engine_model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clip/scores", response_model=ClipScoresResponse)
async def get_clip_scores(
    texts: str = Form(..., description="Comma-separated list of texts to compare with image"),
    image: UploadFile = File(..., description="Image file")
) -> ClipScoresResponse:
    """Calculate CLIP similarity scores between an image and multiple texts."""
    try:
        # Load image from upload
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))

        # Parse comma-separated texts
        text_list = [text.strip() for text in texts.split(',')]

        result = await OrchestratorInterface.get_clip_scores(img, text_list)
        if isinstance(result, dict):
            # Prefer model_name from orchestrator; fallback to engine's currently loaded model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.clip_model, "model_name", None)
            except Exception:
                engine_model_name = None
            model_name = result.get("model_name") or engine_model_name
            return ClipScoresResponse(scores=result.get("result"), model_name=model_name)
        else:
            # Fallback: obtain model_name from the currently loaded engine model
            engine_model_name = None
            try:
                engine_model_name = getattr(OrchestratorInterface.engine.clip_model, "model_name", None)
            except Exception:
                engine_model_name = None
            return ClipScoresResponse(scores=result, model_name=engine_model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ============================================================================
# Clustering Algorithm Endpoints
# ============================================================================

class KMeansRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    n_clusters: int = Field(..., description="Number of clusters")


class ClusteringResponse(BaseModel):
    result: List[List[str]] = Field(..., description="List of clusters, each containing list of texts")


@router.post("/algorithm/k-means", response_model=ClusteringResponse)
async def k_means_clustering(request: KMeansRequest) -> ClusteringResponse:
    """Perform K-Means clustering on data."""
    try:
        result = await OrchestratorInterface.k_means(request.data, request.n_clusters)
        return ClusteringResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AgglomerativeRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    n_clusters: int = Field(..., description="Number of clusters")
    linkage: str = Field(default="ward", description="Linkage method: ward, complete, average, single")


@router.post("/algorithm/agglomerative", response_model=ClusteringResponse)
async def agglomerative_clustering(request: AgglomerativeRequest) -> ClusteringResponse:
    """Perform Agglomerative clustering on data."""
    try:
        result = await OrchestratorInterface.agglomerative(
            request.data,
            request.n_clusters,
            request.linkage
        )
        return ClusteringResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AutoAgglomerativeRequest(BaseModel):
    data: List[dict] = Field(..., description="Data points with 'id' and 'embedding' fields")
    max_clusters: int = Field(..., description="Maximum number of clusters to evaluate")
    linkage: str = Field(default="ward", description="Linkage method: ward, complete, average, single")


@router.post("/algorithm/auto-agglomerative", response_model=ClusteringResponse)
async def auto_agglomerative_clustering(request: AutoAgglomerativeRequest) -> ClusteringResponse:
    """Perform Agglomerative clustering with automatic cluster number selection."""
    try:
        result = await OrchestratorInterface.auto_agglomerative(
            request.data,
            request.max_clusters,
            request.linkage
        )
        return ClusteringResponse(result=result)
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


class BucketItem(BaseModel):
    bucket: str = Field(..., description="Bucket name")
    sentences: List[str] = Field(..., description="List of sentences in this bucket")


class SimilarityBucketingResponse(BaseModel):
    result: List[BucketItem] = Field(..., description="List of buckets with their assigned sentences")


@router.post("/algorithm/similarity-bucketing", response_model=SimilarityBucketingResponse)
async def similarity_bucketing(request: SimilarityBucketingRequest) -> SimilarityBucketingResponse:
    """Perform similarity-based bucketing on data."""
    try:
        result = await OrchestratorInterface.similarity_bucketing(
            request.data,
            request.buckets,
            request.score_method,
            request.allow_orphan_bucket,
            request.orphan_threshold
        )
        buckets = [BucketItem(bucket=item["bucket"], sentences=item["sentences"]) for item in result]
        return SimilarityBucketingResponse(result=buckets)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
