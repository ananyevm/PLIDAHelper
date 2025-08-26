import os
import pickle
from .models import ModelManager

class EmbeddingManager:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def precompute_embeddings(self, df, column, output_path):
        """Precompute and cache embeddings for a DataFrame column."""
        if not os.path.exists(output_path):
            embeddings = self.model_manager.encode(df[column].tolist())
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
        else:
            with open(output_path, 'rb') as f:
                embeddings = pickle.load(f)
        return embeddings
    
    def encode_query(self, query):
        """Encode a single query string."""
        return self.model_manager.encode(query)
    
    def encode_batch(self, texts):
        """Encode a batch of texts."""
        return self.model_manager.encode(texts)