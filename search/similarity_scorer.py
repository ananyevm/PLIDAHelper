"""
Advanced similarity scoring module for semantic search.
Implements multi-factor ranking with adaptive scoring mechanisms.
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter

class AdvancedSimilarityScorer:
    """Advanced scoring system that considers multiple factors for relevance."""
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.importance_weights = {
            'semantic_similarity': 0.6,
            'exact_term_match': 0.2,
            'dataset_relevance': 0.1,
            'variable_quality': 0.1
        }
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords for boosting."""
        return {
            'education': [
                'education', 'educational', 'school', 'university', 'degree', 'qualification',
                'attainment', 'academic', 'learning', 'literacy', 'graduation', 'tertiary'
            ],
            'employment': [
                'employment', 'job', 'work', 'occupation', 'career', 'labor', 'labour',
                'workforce', 'jobseeker', 'unemployed', 'employed', 'worker'
            ],
            'income': [
                'income', 'salary', 'wage', 'earnings', 'financial', 'economic', 'money',
                'payment', 'allowance', 'benefit', 'welfare', 'pension'
            ],
            'health': [
                'health', 'medical', 'hospital', 'disease', 'condition', 'treatment',
                'healthcare', 'clinical', 'patient', 'therapy', 'medication'
            ],
            'demographics': [
                'age', 'gender', 'sex', 'location', 'state', 'region', 'indigenous',
                'ethnicity', 'population', 'household', 'family', 'resident'
            ]
        }
    
    def compute_enhanced_score(self, query: str, result: Dict[str, Any], 
                             base_similarity: float, topic: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute enhanced similarity score using multiple factors.
        
        Args:
            query: Original search query
            result: Result containing 'row' with variable data
            base_similarity: Base cosine similarity score
            topic: Domain topic for context-specific scoring
            
        Returns:
            Tuple of (final_score, score_breakdown)
        """
        row = result['row']
        variable_name = str(row.get('variable_name', '')).lower()
        description = str(row.get('description', '')).lower()
        dataset_name = str(row.get('dataset', '')).lower()
        
        # Score components
        scores = {
            'semantic_similarity': base_similarity,
            'exact_term_match': self._compute_exact_match_score(query, variable_name, description),
            'dataset_relevance': self._compute_dataset_relevance(query, dataset_name, topic),
            'variable_quality': self._compute_variable_quality_score(variable_name, description)
        }
        
        # Apply domain-specific boosts
        if topic:
            domain_boost = self._compute_domain_boost(query, variable_name, description, topic)
            scores['semantic_similarity'] *= (1 + domain_boost)
        
        # Apply adaptive threshold based on query complexity
        adaptive_factor = self._compute_adaptive_factor(query)
        scores['semantic_similarity'] *= adaptive_factor
        
        # Compute weighted final score
        final_score = sum(
            scores[component] * self.importance_weights[component]
            for component in scores
        )
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, scores
    
    def _compute_exact_match_score(self, query: str, variable_name: str, description: str) -> float:
        """Compute score based on exact term matches."""
        query_terms = set(self._tokenize(query.lower()))
        variable_terms = set(self._tokenize(variable_name))
        description_terms = set(self._tokenize(description))
        
        # Remove noise words
        noise_words = {'of', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from'}
        query_terms -= noise_words
        variable_terms -= noise_words
        description_terms -= noise_words
        
        if not query_terms:
            return 0.0
        
        # Score exact matches in variable name (highest weight)
        var_matches = len(query_terms.intersection(variable_terms))
        var_score = var_matches / len(query_terms) * 1.0
        
        # Score exact matches in description (medium weight)
        desc_matches = len(query_terms.intersection(description_terms))
        desc_score = desc_matches / len(query_terms) * 0.7
        
        # Bonus for complete query phrase in description
        phrase_bonus = 0.0
        if len(query.split()) > 1 and query.lower() in description:
            phrase_bonus = 0.3
        
        return min(1.0, var_score + desc_score + phrase_bonus)
    
    def _compute_dataset_relevance(self, query: str, dataset_name: str, topic: str) -> float:
        """Compute relevance based on dataset characteristics."""
        score = 0.0
        
        # Boost for datasets that match query domain
        if topic:
            topic_keywords = self.domain_keywords.get(topic, [])
            for keyword in topic_keywords:
                if keyword in dataset_name:
                    score += 0.1
        
        # Boost for comprehensive datasets (Census, ATO, etc.)
        comprehensive_datasets = ['census', 'ato', 'core', 'alcd']
        if any(term in dataset_name for term in comprehensive_datasets):
            score += 0.2
        
        # Boost for recent/current datasets
        if any(year in dataset_name for year in ['2021', '2020', '2019', 'current']):
            score += 0.1
        
        return min(1.0, score)
    
    def _compute_variable_quality_score(self, variable_name: str, description: str) -> float:
        """Compute score based on variable quality indicators."""
        score = 0.5  # Base score
        
        # Penalize very short or unclear descriptions
        if len(description) < 10:
            score -= 0.2
        elif len(description) > 30:
            score += 0.1
        
        # Boost for clear, descriptive variable names
        if len(variable_name) > 3 and not variable_name.replace('_', '').isalnum():
            score += 0.1
        
        # Penalize overly technical variable names without context
        if len(variable_name.split('_')) > 4 and len(description) < 20:
            score -= 0.1
        
        # Boost for variables with year information (indicates temporal data)
        if re.search(r'\b(20\d{2}|19\d{2})\b', description + ' ' + variable_name):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compute_domain_boost(self, query: str, variable_name: str, 
                            description: str, topic: str) -> float:
        """Compute domain-specific boost factor."""
        if topic not in self.domain_keywords:
            return 0.0
        
        domain_terms = self.domain_keywords[topic]
        combined_text = f"{query} {variable_name} {description}".lower()
        
        # Count domain-relevant terms
        domain_matches = sum(1 for term in domain_terms if term in combined_text)
        
        # Boost factor based on domain relevance
        boost = min(0.2, domain_matches * 0.05)
        
        return boost
    
    def _compute_adaptive_factor(self, query: str) -> float:
        """Compute adaptive factor based on query complexity."""
        query_terms = self._tokenize(query.lower())
        
        # Simple queries get less strict scoring
        if len(query_terms) <= 2:
            return 1.1
        
        # Complex queries get more strict scoring
        elif len(query_terms) >= 5:
            return 0.95
        
        # Medium complexity
        return 1.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, handling underscores and special chars."""
        # Replace underscores with spaces and split
        text = re.sub(r'[_\-]', ' ', text)
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 1]
    
    def explain_score(self, query: str, result: Dict[str, Any], 
                     final_score: float, score_breakdown: Dict[str, float]) -> str:
        """Generate human-readable explanation of the scoring."""
        row = result['row']
        explanation = [
            f"Variable: {row.get('variable_name', 'N/A')}",
            f"Final Score: {final_score:.3f}",
            f"",
            f"Score Breakdown:",
            f"  Semantic Similarity: {score_breakdown['semantic_similarity']:.3f} (weight: {self.importance_weights['semantic_similarity']})",
            f"  Exact Term Match: {score_breakdown['exact_term_match']:.3f} (weight: {self.importance_weights['exact_term_match']})",
            f"  Dataset Relevance: {score_breakdown['dataset_relevance']:.3f} (weight: {self.importance_weights['dataset_relevance']})",
            f"  Variable Quality: {score_breakdown['variable_quality']:.3f} (weight: {self.importance_weights['variable_quality']})"
        ]
        
        return "\n".join(explanation)