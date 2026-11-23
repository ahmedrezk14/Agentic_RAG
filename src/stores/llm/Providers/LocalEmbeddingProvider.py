import logging
from ..LLMInterface import LLMInterface

class LocalEmbeddingProvider(LLMInterface):

    def __init__(self,default_input_max_characters: int=1000, device="cpu"):
        self.default_input_max_characters=default_input_max_characters
        self.device=device
        self.generation_model_id=None
        self.embedding_model_id=None
        self.embedding_size=None
        self._model=None
        self.logger=logging.getLogger(__name__)
        
    def set_generation_model(self, model_id: str):
        self.generation_model_id=model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id=model_id
        self.embedding_size=embedding_size

    def generate_text(self, prompt: str,chat_history: list=[], max_output_token: int=None, temperature: float=None):
        raise NotImplementedError("LocalEmbeddingProvider does not support text generation.")

    def _load_model(self):
        if self._model is None:
            if not self.embedding_model_id:
                raise ValueError("Embedding model ID is not configured.")
            from sentence_transformers import SentenceTransformer
            self._model=SentenceTransformer(self.embedding_model_id,device=self.device)
        return self._model
            


    def embed_text(self, text: str, document_type:str=None):
        
        if isinstance(text, str):
            text=[text]
        
        processed=[self.process_text(t) for t in text]
        model=self._load_model()
        embeddings=model.encode(
            processed,
            normalize_embeddings=True,
            show_progress_bar=False
        ) # return numpy array
        
        return [vec.tolist() for vec in embeddings]

    def construct_prompt(self, prompt:str, role:str):
        return {
            "role":role,
            "content":prompt
        }

    def process_text(self, text:str):
        return text[:self.default_input_max_characters].strip()