from pathlib import Path

# Global model registry
model_map = {}


def register_model(cls):
    """
    Decorator to register a model class.
    Uses the model_id class attribute to register the model.
    """
    if hasattr(cls, 'model_id'):
        model_map[cls.model_id] = cls
    return cls


class ModelLoader:
    """Acts like a factory for models"""

    CACHE_DIR = Path(r"E:\model_cache")

    @classmethod
    def load_model(cls, model_name: str, device: str = "cpu"):
        """
        Load a model using the registered model class.

        Args:
            model_name: HuggingFace model ID
            device: Device to load the model on

        Returns:
            Instance of the registered model wrapper class
        """
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if model_name not in model_map:
            raise ValueError(
                f"Model '{model_name}' is not registered. "
                f"Available models: {list(model_map.keys())}"
            )

        wrapper_class = model_map[model_name]
        wrapper_instance = wrapper_class()
        wrapper_instance.initialize(model_name, device, str(cls.CACHE_DIR))

        return wrapper_instance
