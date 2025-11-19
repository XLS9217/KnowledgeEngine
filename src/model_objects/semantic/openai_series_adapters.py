import torch
from PIL.Image import Image
from transformers import CLIPModel, CLIPProcessor
from src.model_objects.model_loader import register_model
from src.model_objects.model_bases import CLIPModelBase


@register_model
class CLIPVITBasePatch32(CLIPModelBase):
    model_id = "openai/clip-vit-base-patch32"

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True
        )

    def get_clip_score(self, img: Image, text: str):
        """Get similarity score between image and text."""
        inputs = self.processor(text=[text], images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits_per_image.item()
        return score

    def get_clip_scores(self, img: Image, texts: list[str]):
        """Get similarity scores between image and multiple texts. Returns list of (text, score)."""
        inputs = self.processor(text=texts, images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        return [(text, prob.item()) for text, prob in zip(texts, probs)]