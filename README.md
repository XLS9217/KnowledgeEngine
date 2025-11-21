# Knowledge Engine

A FastAPI-based NLP task engine that provides low-level NLP services using local models, including text embeddings, reranking, CLIP (vision-language), and clustering/bucketing algorithms.

## Features

- **Text Embeddings**: Generate semantic embeddings for single or multiple texts
- **Reranking**: Rerank documents based on query relevance
- **CLIP**: Calculate image-text similarity scores using vision-language models
- **Clustering Algorithms**: K-Means, Agglomerative, and Auto-Agglomerative clustering
- **Similarity Bucketing**: Organize data into predefined buckets based on semantic similarity
- **Local Model Execution**: All models run locally with GPU acceleration (CUDA 12.6)

## Prerequisites

- Python 3.11+
- CUDA 12.6 (for GPU acceleration)
- Local model cache at `E:\model_cache` (models will be loaded from here first)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install dependencies
uv sync
```

## Running the Server

```bash
# Start the FastAPI server
uv run -m main
```

The server will start on `http://0.0.0.0:7009`

## API Endpoints

### Embedding Services

- `POST /api/v1/embedding` - Generate embedding for a single text
- `POST /api/v1/embeddings` - Generate embeddings for multiple texts

### Reranking Services

- `POST /api/v1/rerank` - Rerank documents based on query relevance

### CLIP Services

- `POST /api/v1/clip/score` - Calculate similarity between an image and text
- `POST /api/v1/clip/scores` - Calculate similarity between an image and multiple texts

### Clustering Algorithms

- `POST /api/v1/algorithm/k-means` - K-Means clustering
- `POST /api/v1/algorithm/agglomerative` - Agglomerative clustering
- `POST /api/v1/algorithm/auto-agglomerative` - Auto-tuned agglomerative clustering

### Bucketing Algorithms

- `POST /api/v1/algorithm/similarity-bucketing` - Similarity-based bucketing

## Testing

Run example scripts to test different functionalities:

```bash
# Test embedding models
uv run examples/mod/ebd_t.py
uv run examples/mod/ebds_t.py

# Test reranking
uv run examples/mod/rerank_t.py

# Test CLIP
uv run examples/mod/clip_t.py

# Test clustering algorithms
uv run examples/alg/clus_t.py

# Test bucketing algorithms
uv run examples/alg/simbuk_t.py

# Test HTTP endpoints
uv run examples/http/ebd_h.py
uv run examples/http/rerank_h.py
uv run examples/http/clip_h.py
uv run examples/http/bth_clus_h.py
uv run examples/http/simbuk_h.py

# Test orchestrator engine
uv run examples/orch/spe_erc_t.py
uv run examples/orch/spe_ebds_km.py
uv run examples/orch/spe_alg_t.py
```

## Project Structure

```
main.py                                    # FastAPI application entry point
pyproject.toml                             # Project dependencies and configuration
├── src/
│   ├── algorisms/                         # Algorithm implementations
│   │   ├── clustering.py                  # K-Means, Agglomerative clustering
│   │   ├── bucketing.py                   # Similarity-based bucketing
│   │   └── algorism_structs.py            # Data structures for algorithms
│   ├── model_objects/                     # Model adapters and loaders
│   │   ├── model_loader.py                # Load and manage ML models
│   │   ├── model_bases.py                 # Base classes for model adapters
│   │   └── semantic/                      # Semantic model adapters
│   │       ├── jina_series_adapters.py    # Jina embedding models
│   │       ├── qwen_series_adapters.py    # Qwen embedding models
│   │       └── openai_series_adapters.py  # OpenAI-compatible models
│   ├── routers/                           # FastAPI route handlers
│   │   ├── service_router.py              # Main service endpoints
│   │   └── system_router.py               # System/health endpoints
│   ├── task_orchestrator/                 # Task execution engine
│   │   ├── orchestrator_interface.py      # Public interface for task execution
│   │   ├── orchestrator_engine_base.py    # Base engine implementation
│   │   ├── engine_structs.py              # Task request/response structures
│   │   ├── single_process_engine/         # Single-process execution engine
│   │   │   └── engine.py                  # Synchronous task executor
│   │   └── asyncio_ray_engine/            # Distributed execution engine (future)
│   │       └── engine.py                  # Ray-based distributed executor
│   └── utils/                             # Utility modules
│       ├── setup_logging.py               # Logging configuration
│       └── device_watcher.py              # GPU/device monitoring
├── examples/                              # Example scripts and tests
│   ├── mod/                               # Model testing scripts
│   ├── alg/                               # Algorithm testing scripts
│   ├── http/                              # HTTP endpoint testing scripts
│   ├── orch/                              # Orchestrator testing scripts
│   └── test_data/                         # Test data and utilities
└── .env                                   # Environment configuration
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                      (main.py)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   System    │ │   Service   │ │  Future     │
│   Router    │ │   Router    │ │  Routers    │
└──────┬──────┘ └──────┬──────┘ └─────────────┘
       │               │
       │               ▼
       │      ┌────────────────────┐
       │      │ Orchestrator       │
       │      │ Interface          │
       │      └────────┬───────────┘
       │               │
       │      ┌────────┴───────────┐
       │      │                    │
       │      ▼                    ▼
       │  ┌──────────────┐  ┌──────────────┐
       │  │ Single       │  │ AsyncIO+Ray  │
       │  │ Process      │  │ Engine       │
       │  │ Engine       │  │ (Future)     │
       │  └──────┬───────┘  └──────────────┘
       │         │
       │  ┌──────┴─────────────────────┐
       │  │                            │
       │  ▼                            ▼
       │ ┌──────────┐          ┌──────────────┐
       │ │ Model    │          │ Algorithm    │
       │ │ Objects  │          │ Executors    │
       │ └────┬─────┘          └──────────────┘
       │      │
       │  ┌───┴────────────────┐
       │  │                    │
       │  ▼                    ▼
       │ ┌────────┐      ┌────────────┐
       │ │Embedding│      │   CLIP     │
       │ │ Models │      │  Reranker  │
       │ └────────┘      └────────────┘
       │
       ▼
 ┌─────────────┐
 │  Logging &  │
 │  Utilities  │
 └─────────────┘
```

## Model Management

- Models are cached locally in `E:\model_cache`
- The system checks this directory first before downloading from HuggingFace Hub
- Supported model types:
  - Embedding models (Jina, Qwen series)
  - Reranking models
  - CLIP models (vision-language)

## Development Guide

1. **Adding New Endpoints**: Create new routes in `src/routers/`
2. **Adding New Models**: Implement adapters in `src/model_objects/semantic/`
3. **Adding New Algorithms**: Implement in `src/algorisms/`
4. **Task Orchestration**: Use `OrchestratorInterface` for task execution
5. **Testing**: Add example scripts in `examples/` directory

## Configuration

Environment variables can be set in `.env` file:
- Model cache location
- API keys (if using external APIs)
- Server configuration

## License

[License information to be added]
