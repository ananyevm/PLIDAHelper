import numpy as np
from config import DEFAULT_TOP_K, MIN_SCORE_THRESHOLD
from embeddings import EmbeddingManager

class SemanticSearchEngine:
    def __init__(self, indices, dataframes):
        self.indices = indices
        self.dataframes = dataframes
        self.embedding_manager = EmbeddingManager()
    
    def search_datasets(self, query, top_k=DEFAULT_TOP_K):
        """Search for relevant datasets."""
        return self._search('datasets', query, top_k, boost_datasets=None)
    
    def search_variables(self, query, index_name='variables', top_k=DEFAULT_TOP_K, boost_datasets=None):
        """Search for relevant variables in a specific index.
        
        Args:
            query: Search query
            index_name: Name of index to search
            top_k: Number of results to return
            boost_datasets: List of dataset names to prioritize in results
        """
        return self._search(index_name, query, top_k, boost_datasets=boost_datasets)
    
    def _search(self, index_name, query, top_k, boost_datasets=None):
        """Perform semantic search on a specific index."""
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' not found")
        
        # Encode query
        query_embedding = self.embedding_manager.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        index = self.indices[index_name]
        # Increase search size to account for filtering and boosting
        search_k = min(top_k * 3, len(self.dataframes[index_name]))
        D, I = index.search(query_embedding, search_k)
        
        # Convert distances to scores
        scores = 1 - D[0] / 2
        indices = I[0]
        
        # Filter by threshold and prepare results
        results = []
        df = self.dataframes[index_name]
        
        for idx, score in zip(indices, scores):
            if score > MIN_SCORE_THRESHOLD:
                row = df.iloc[idx]
                final_score = score
                
                # Apply boost if this variable is from a selected dataset
                if boost_datasets and 'dataset' in row:
                    # Check if the variable's dataset matches any of the selected datasets
                    for dataset_name in boost_datasets:
                        # Match on dataset name or ID (handling case differences)
                        if (dataset_name.lower() in str(row['dataset']).lower() or 
                            str(row['dataset']).lower() in dataset_name.lower()):
                            # Apply a 30% boost to prioritize these variables
                            final_score = min(score * 1.3, 1.0)
                            break
                
                results.append({
                    'row': row,
                    'score': final_score,
                    'index': idx,
                    'original_score': score,
                    'boosted': final_score != score
                })
        
        # Sort by final score and return top_k results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def deduplicate_results(self, results, key_field='description'):
        """Remove duplicate results based on a key field."""
        seen = {}
        deduplicated = []
        
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            key_value = result['row'][key_field]
            if key_value not in seen:
                seen[key_value] = True
                deduplicated.append(result)
        
        return deduplicated