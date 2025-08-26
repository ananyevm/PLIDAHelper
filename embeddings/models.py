from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class ModelManager:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get or initialize the SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._model
    
    def encode(self, texts, **kwargs):
        """Encode texts to embeddings."""
        model = self.get_model()
        return model.encode(texts, convert_to_numpy=True, **kwargs)