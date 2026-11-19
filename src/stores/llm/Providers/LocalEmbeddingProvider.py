import logging
from typing import List, Union

from ..LLMInterface import LLMInterface


class LocalEmbeddingProvider(LLMInterface):
    """Local embedding provider using sentence-transformers on CPU."""

    def __init__(self,default_input_max_characters: int = 1000,device: str = "cpu",) -> None:
        self.default_input_max_characters = default_input_max_characters
        self.device = device
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        self._model = None
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def _get_model(self):
        """Lazily load the SentenceTransformer model on first use."""
        if self._model is None:
            if not self.embedding_model_id:
                raise ValueError("Embedding model ID is not configured.")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_id, device=self.device)
        return self._model

    def generate_text(self,prompt: str,chat_history: list = [],max_output_token: int = None,temperature: float = None,):
        raise NotImplementedError("LocalEmbeddingProvider does not support text generation.")

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        """Encode a string or list of strings into L2-normalized embedding vectors."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        processed = [self.process_text(t) for t in texts]
        model = self._get_model()
        embeddings = model.encode(
            processed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in embeddings]

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": prompt}

    def process_text(self, text: str) -> str:
        return text[: self.default_input_max_characters].strip()
