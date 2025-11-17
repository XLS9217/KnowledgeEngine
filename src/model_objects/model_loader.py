from pathlib import Path
from transformers import AutoModel


# Global model registry
model_map = {}


def register_model(cls):
    """
    Decorator to register a model class.
    Uses the model_id class attribute to register the model.

    Example:
        @register_model
        class JinaEmbeddingsV3(EmbeddingModelBase):
            model_id = "jinaai/jina-embeddings-v3"
    """
    if hasattr(cls, 'model_id'):
        model_map[cls.model_id] = cls
    return cls



class ModelLoader:

    """
    Acts like a factory for models
    """

    # Define cache directory relative to project root
    CACHE_DIR = Path(__file__).parent.parent.parent / "model_cache"

    @classmethod
    def load_model(
            cls,
            model_name: str,
            force_local_only: bool = False,
            device: str = "cpu"
    ):
        """
        Load a model using the registered model class.

        Args:
            model_name: HuggingFace model ID (e.g., "jinaai/jina-embeddings-v3")
            force_local_only: If True, only load from local cache (no download)
            device: Device to load the model on (e.g., "cpu", "cuda:0", "mps")

        Returns:
            Instance of the registered model wrapper class

        Raises:
            ValueError: If model is not registered
        """
        # Ensure cache directory exists
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check if model is registered
        if model_name not in model_map:
            raise ValueError(
                f"Model '{model_name}' is not registered. "
                f"Available models: {list(model_map.keys())}"
            )

        # Get the wrapper class from registry
        wrapper_class = model_map[model_name]

        # Load the HuggingFace model
        hf_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cls.CACHE_DIR),
            local_files_only=force_local_only,
            trust_remote_code=True
        )

        # Move to device
        hf_model = hf_model.to(device)

        # Create wrapper instance with device parameter
        wrapper_instance = wrapper_class(device=device)
        wrapper_instance.model = hf_model

        return wrapper_instance
