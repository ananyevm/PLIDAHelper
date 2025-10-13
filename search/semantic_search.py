import numpy as np
from typing import List, Dict, Any, Tuple
from config import DEFAULT_TOP_K, MIN_SCORE_THRESHOLD
from embeddings import EmbeddingManager
from .query_expansion import QueryExpansion
from .similarity_scorer import AdvancedSimilarityScorer
from .relevance_scorer import RelevanceScorer

class SemanticSearchEngine:
    def __init__(self, indices, dataframes, llm_client=None):
        self.indices = indices
        self.dataframes = dataframes
        self.embedding_manager = EmbeddingManager()
        self.query_expansion = QueryExpansion(llm_client) if llm_client else QueryExpansion()
        self.similarity_scorer = AdvancedSimilarityScorer()
        self.relevance_scorer = RelevanceScorer(llm_client)
    
    def search_datasets(self, query, top_k=DEFAULT_TOP_K):
        """Search for relevant datasets."""
        return self._search('datasets', query, top_k, boost_datasets=None)
    
    def search_variables(self, query, index_name='variables', top_k=DEFAULT_TOP_K, boost_datasets=None, topic=None, use_expansion=True, use_advanced_scoring=True, use_openai_relevance=True, conceptual_variable=None):
        """Search for relevant variables in a specific index.
        
        Args:
            query: Search query
            index_name: Name of index to search
            top_k: Number of results to return
            boost_datasets: List of dataset names to prioritize in results
            topic: Research topic for domain-specific enhancement
            use_expansion: Whether to use query expansion
            use_advanced_scoring: Whether to use advanced multi-factor scoring
            use_openai_relevance: Whether to use OpenAI to score variable relevance
            conceptual_variable: Description of conceptual variable needed (for OpenAI scoring)
        """
        # Get search results
        if use_expansion and topic:
            results = self._enhanced_search(index_name, query, top_k, boost_datasets, topic, use_advanced_scoring)
        else:
            results = self._search(index_name, query, top_k, boost_datasets=boost_datasets, topic=topic, use_advanced_scoring=use_advanced_scoring)
        
        # Apply OpenAI relevance scoring if requested
        if use_openai_relevance and results:
            # Use conceptual_variable if provided, otherwise use the query
            concept = conceptual_variable if conceptual_variable else query
            results = self._add_openai_relevance_scores(results, concept)
        
        return results
    
    def _search(self, index_name, query, top_k, boost_datasets=None, topic=None, use_advanced_scoring=False):
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
                result_data = {'row': row}
                
                # Use advanced scoring if enabled
                if use_advanced_scoring:
                    final_score, score_breakdown = self.similarity_scorer.compute_enhanced_score(
                        query, result_data, score, topic
                    )
                    
                    # Apply dataset boost to the final score if needed
                    boost_applied = False
                    if boost_datasets and 'dataset' in row:
                        for dataset_name in boost_datasets:
                            if (dataset_name.lower() in str(row['dataset']).lower() or 
                                str(row['dataset']).lower() in dataset_name.lower()):
                                final_score = min(final_score * 1.1, 1.0)  # Smaller boost since advanced scoring already considers dataset relevance
                                boost_applied = True
                                break
                    
                    results.append({
                        'row': row,
                        'score': final_score,
                        'index': idx,
                        'original_score': score,
                        'boosted': boost_applied,
                        'advanced_scoring': True,
                        'score_breakdown': score_breakdown
                    })
                else:
                    # Use simple scoring (original behavior)
                    final_score = score
                    
                    # Apply boost if this variable is from a selected dataset
                    boost_applied = False
                    if boost_datasets and 'dataset' in row:
                        for dataset_name in boost_datasets:
                            if (dataset_name.lower() in str(row['dataset']).lower() or 
                                str(row['dataset']).lower() in dataset_name.lower()):
                                final_score = min(score * 1.3, 1.0)
                                boost_applied = True
                                break
                    
                    results.append({
                        'row': row,
                        'score': final_score,
                        'index': idx,
                        'original_score': score,
                        'boosted': boost_applied,
                        'advanced_scoring': False
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
    
    def _enhanced_search(self, index_name: str, query: str, top_k: int, boost_datasets: List[str], topic: str, use_advanced_scoring: bool = True) -> List[Dict[str, Any]]:
        """Perform enhanced search with query expansion."""
        # Generate query variants
        variants = self.query_expansion.generate_query_variants(query, topic)
        weights = self.query_expansion.get_expansion_weights()
        
        all_results = []
        
        # Search with each variant
        for variant_type, variant_query in variants:
            try:
                # Get weight for this variant type
                weight = weights.get(variant_type, 0.5)
                
                # Perform search with advanced scoring
                results = self._search(index_name, variant_query, top_k * 2, boost_datasets, topic, use_advanced_scoring)
                
                # Apply weight and mark expansion type
                for result in results:
                    result['score'] *= weight
                    result['expansion_type'] = variant_type
                    result['original_query'] = query
                    result['expanded_query'] = variant_query
                    all_results.append(result)
                    
            except Exception:
                # Skip failed searches
                continue
        
        # Combine and deduplicate results
        combined_results = self._combine_expansion_results(all_results)
        
        # Filter out irrelevant results
        filtered_results = self._filter_irrelevant_results(combined_results, query)
        
        # Sort by final score and return top results
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        return filtered_results[:top_k]
    
    def _combine_expansion_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results from different query expansions with intelligent deduplication."""
        # Group results by variable name and dataset
        grouped = {}
        
        for result in results:
            row = result['row']
            # Create key based on variable name and dataset
            key = (
                str(row.get('variable_name', '')).strip().lower(),
                str(row.get('dataset_id', '')).strip().upper()
            )
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # For each group, select the best result
        final_results = []
        for key, group_results in grouped.items():
            # Sort by score and take the best one
            best_result = max(group_results, key=lambda x: x['score'])
            
            # Add information about how many expansion types found this result
            expansion_types = list(set(r['expansion_type'] for r in group_results))
            best_result['found_in_expansions'] = expansion_types
            best_result['expansion_count'] = len(expansion_types)
            
            # Boost score if found in multiple expansions (indicates higher relevance)
            if len(expansion_types) > 1:
                best_result['score'] *= 1.1  # 10% boost for multi-expansion matches
            
            final_results.append(best_result)
        
        return final_results
    
    def _filter_irrelevant_results(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Filter out results that are clearly irrelevant to the original query."""
        filtered_results = []
        original_query_lower = original_query.lower()
        
        for result in results:
            row = result['row']
            variable_name = str(row.get('variable_name', '')).lower()
            description = str(row.get('description', '')).lower()
            combined_text = f"{variable_name} {description}"
            
            should_exclude = False
            
            # Filter out marital status for employment queries
            if 'employment' in original_query_lower:
                if any(term in combined_text for term in ['marital', 'marriage', 'spouse', 'partner']):
                    should_exclude = True
            
            # Filter out employment terms for marital queries
            elif any(term in original_query_lower for term in ['marital', 'marriage']):
                if any(term in combined_text for term in ['employment', 'job', 'work', 'occupation']):
                    should_exclude = True
            
            # Filter out marital terms for poverty queries
            elif 'poverty' in original_query_lower:
                if any(term in combined_text for term in ['marital', 'marriage', 'spouse']):
                    should_exclude = True
            
            # Filter out income derivation indicators for education queries
            elif any(term in original_query_lower for term in ['education', 'educational', 'attainment']):
                if any(term in combined_text for term in ['income derivation', 'derivation indicator', 'household income derivation', 'family income derivation']):
                    should_exclude = True
            
            if not should_exclude:
                filtered_results.append(result)
        
        return filtered_results
    
    def search_with_concept_detection(self, query: str, index_name: str = 'variables', 
                                    top_k: int = DEFAULT_TOP_K, boost_datasets: List[str] = None, 
                                    topic: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Search with automatic concept detection and expansion."""
        # Detect concepts in the query
        detected_concepts = self.query_expansion.detect_query_concepts(query)
        
        # Perform enhanced search
        results = self._enhanced_search(index_name, query, top_k, boost_datasets, topic or 'general')
        
        # Add concept-specific variable suggestions if no good results
        if len(results) < 3 and detected_concepts:
            for concept in detected_concepts:
                concept_vars = self.query_expansion.get_context_specific_variables(concept)
                for var in concept_vars[:2]:  # Add top 2 concept-specific variables
                    concept_results = self._search(index_name, var, 2, boost_datasets)
                    for result in concept_results:
                        result['expansion_type'] = 'concept_suggestion'
                        result['suggested_for_concept'] = concept
                        results.append(result)
        
        # Final deduplication and sorting
        final_results = self._combine_expansion_results(results)
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k], detected_concepts
    
    def _add_openai_relevance_scores(self, results: List[Dict[str, Any]], conceptual_variable: str) -> List[Dict[str, Any]]:
        """Add OpenAI relevance scores to search results."""
        if not results:
            return results
        
        # Extract variable data for scoring
        variables_to_score = []
        for result in results:
            row = result['row']
            variable_data = {
                'variable_name': row.get('variable_name', ''),
                'description': row.get('description', ''),
                'dataset': row.get('dataset', '')
            }
            variables_to_score.append(variable_data)
        
        # Score variables using batch processing for efficiency
        try:
            relevance_scores = self.relevance_scorer.score_batch_relevance(
                conceptual_variable, variables_to_score
            )
        except Exception as e:
            # Return default scores for all variables if OpenAI fails
            relevance_scores = [(0.5, "Could not evaluate relevance due to API error")] * len(variables_to_score)
        
        # Add relevance scores to results
        for i, result in enumerate(results):
            if i < len(relevance_scores):
                probability, explanation = relevance_scores[i]
                result['openai_relevance'] = probability
                result['relevance_explanation'] = explanation
            else:
                result['openai_relevance'] = 0.5
                result['relevance_explanation'] = "Could not evaluate relevance"
        
        return results
    
    def explain_result_score(self, query: str, result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of how a result was scored."""
        if result.get('advanced_scoring') and 'score_breakdown' in result:
            return self.similarity_scorer.explain_score(
                query, result, result['score'], result['score_breakdown']
            )
        else:
            # Simple explanation for basic scoring
            explanation = [
                f"Variable: {result['row'].get('variable_name', 'N/A')}",
                f"Score: {result['score']:.3f}",
                f"Scoring Method: Basic cosine similarity",
                f"Original Score: {result.get('original_score', result['score']):.3f}"
            ]
            
            if result.get('boosted'):
                explanation.append(f"Dataset Boost Applied: Yes")
            
            if result.get('expansion_type'):
                explanation.append(f"Query Expansion: {result['expansion_type']}")
            
            return "\n".join(explanation)