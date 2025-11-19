import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from src.model_objects import register_model
from src.model_objects.model_bases import EmbeddingModelBase, RerankerModelBase


@register_model
class Qwen3Embedding06B(EmbeddingModelBase):
    model_id = "Qwen/Qwen3-Embedding-0.6B"

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)

    def get_embedding(self, text: str):
        embeddings = self.model.encode(text)
        return embeddings


@register_model
class Qwen3Reranker06B(RerankerModelBase):
    model_id = "Qwen/Qwen3-Reranker-0.6B"

    def initialize(self, model_name: str, device: str, model_path: str):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_path,
            local_files_only=True,
            trust_remote_code=True
        )

    def rerank(self, query: str, documents: list[str], top_k: int = 5):
        """Rerank documents by relevance to query. Returns list of (index, score, document)."""
        instruction = "Given a query, retrieve relevant passages that answer the query"
        prefix = "<|prefix|>yes<|suffix|>"
        suffix = "<|endoftext|>"

        scores = []
        for doc in documents:
            prompt = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = prefix + text + suffix

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]

            yes_id = self.tokenizer.convert_tokens_to_ids("yes")
            no_id = self.tokenizer.convert_tokens_to_ids("no")
            score = torch.softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], dim=-1), dim=-1)[:, 1]
            scores.append(score.item())

        results = [(i, scores[i], documents[i]) for i in range(len(documents))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
