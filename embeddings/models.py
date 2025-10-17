from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, SEED

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
            # Set random seed for deterministic embeddings
            if hasattr(self._model, 'set_seed'):
                self._model.set_seed(SEED)
        return self._model
    
    def encode(self, texts, **kwargs):
        """Encode texts to embeddings."""
        model = self.get_model()
        # Ensure deterministic behavior by setting the seed before each encoding
        import torch
        import numpy as np
        import random
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        return model.encode(texts, convert_to_numpy=True, **kwargs)