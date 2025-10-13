"""
Query expansion module for improving search relevance through concept mapping,
domain-specific enhancements, and multi-stage search strategies.
"""

import json
from typing import List, Dict, Tuple, Any

class QueryExpansion:
    """Handles intelligent query expansion for better search results."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.concept_mappings = self._initialize_concept_mappings()
        self.domain_enhancements = self._initialize_domain_enhancements()
        self.variable_suggestions = self._initialize_variable_suggestions()
        self.noise_words = self._initialize_noise_words()
    
    def _initialize_concept_mappings(self) -> Dict[str, List[str]]:
        """Initialize concept mapping dictionary."""
        return {
            'poverty': [
                'income', 'financial hardship', 'economic disadvantage', 'low income', 
                'household income', 'family income', 'earnings', 'welfare', 'benefits', 
                'allowance', 'financial stress', 'economic hardship', 'income support'
            ],
            'wealth': [
                'income', 'assets', 'property', 'investments', 'savings', 
                'financial resources', 'high income', 'affluence', 'prosperity'
            ],
            'socioeconomic': [
                'income', 'education', 'occupation', 'employment', 'social class',
                'economic status', 'social status', 'educational attainment'
            ],
            'health outcomes': [
                'mortality', 'morbidity', 'hospital admissions', 'medical conditions', 
                'treatments', 'health status', 'medical procedures', 'health services'
            ],
            'education quality': [
                'graduation rates', 'achievement', 'test scores', 'qualifications', 
                'literacy', 'educational attainment', 'academic performance'
            ],
            'social mobility': [
                'income change', 'occupation change', 'education advancement', 
                'social class movement', 'economic mobility', 'career progression'
            ],
            'unemployment': [
                'employment status', 'jobless', 'labor force status', 'workforce participation',
                'job seeking', 'unemployment benefits', 'jobseeker', 'employment rate'
            ],
            'employment': [
                'job', 'work', 'occupation', 'career', 'labor force', 'workforce',
                'employed', 'working', 'employment rate', 'job type', 'industry'
            ],
            'health': [
                'medical', 'healthcare', 'wellness', 'wellbeing', 'medical conditions',
                'hospital', 'treatment', 'diagnosis', 'health services', 'medical care'
            ],
            'disability': [
                'impairment', 'assistance needed', 'accessibility', 'support services',
                'functional limitation', 'activity restriction', 'participation restriction'
            ],
            'mental health': [
                'psychological wellbeing', 'mental disorders', 'anxiety', 'depression',
                'mental illness', 'psychological distress', 'mental health services'
            ]
        }
    
    def _initialize_domain_enhancements(self) -> Dict[str, List[str]]:
        """Initialize domain-specific enhancement terms."""
        return {
            'poverty': [
                'income quintile', 'household income', 'family income', 'earnings', 
                'welfare payments', 'financial stress', 'housing costs', 'rent burden'
            ],
            'healthcare': [
                'medical conditions', 'hospital utilization', 'treatment outcomes', 
                'health services', 'medical procedures', 'pharmaceutical benefits'
            ],
            'education': [
                'educational attainment', 'qualifications', 'academic achievement', 
                'graduation', 'tertiary education', 'university degree'
            ],
            'employment': [
                'labor force participation', 'occupation', 'industry', 'work hours', 
                'unemployment benefits', 'job type', 'employment type', 'workforce'
            ],
            'immigration': [
                'country of birth', 'migration background', 'citizenship', 'visa type',
                'language spoken', 'year of arrival', 'birthplace'
            ],
            'social services': [
                'welfare receipt', 'government assistance', 'social support', 
                'benefit payments', 'support services', 'community services'
            ]
        }
    
    def _initialize_variable_suggestions(self) -> Dict[str, List[str]]:
        """Initialize concept-specific variable suggestions."""
        return {
            'poverty': [
                'household income quartile', 'family income weekly', 'equivalised household income',
                'welfare payment receipt', 'housing stress indicators', 'unemployment benefits',
                'rental assistance', 'income support payments'
            ],
            'wealth': [
                'household income', 'property ownership', 'investment income', 
                'asset ownership', 'housing value', 'financial investments'
            ],
            'health_outcomes': [
                'hospital admissions', 'medical procedures', 'medication usage',
                'health service utilization', 'medical conditions', 'mortality'
            ],
            'education': [
                'highest qualification', 'university degree', 'educational attainment',
                'field of study', 'graduation year', 'academic achievement'
            ],
            'employment': [
                'employment status', 'occupation type', 'industry sector',
                'work hours', 'employment income', 'labor force participation'
            ],
            'health': [
                'health status', 'medical conditions', 'hospital admissions',
                'health services', 'medical procedures', 'health outcomes'
            ]
        }
    
    def expand_query_concepts(self, query: str) -> str:
        """Expand abstract concepts to concrete measurable variables."""
        expanded_terms = []
        query_lower = query.lower()
        
        for concept, expansions in self.concept_mappings.items():
            if concept in query_lower:
                expanded_terms.extend(expansions[:5])  # Limit to avoid token explosion
        
        # Combine original query with expanded terms
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def enhance_query_by_domain(self, query: str, topic: str) -> str:
        """Add domain-specific terms based on research topic."""
        if topic in self.domain_enhancements:
            enhanced_terms = self.domain_enhancements[topic][:5]  # Limit terms
            return f"{query} {' '.join(enhanced_terms)}"
        return query
    
    def get_context_specific_variables(self, concept: str) -> List[str]:
        """Get pre-defined variables for specific concepts."""
        return self.variable_suggestions.get(concept, [])
    
    def llm_expand_query(self, query: str) -> str:
        """Use LLM to intelligently expand query concepts."""
        if not self.llm_client:
            return query
            
        try:
            prompt = self._build_expansion_prompt(query)
            system_prompt = "You are a research methodology expert. Return only the expanded search terms."
            
            response = self.llm_client.complete(system_prompt, prompt)
            expanded_terms = self._parse_llm_expansion(response)
            
            if expanded_terms:
                return f"{query} {' '.join(expanded_terms)}"
                
        except Exception:
            # Fallback to concept expansion if LLM fails
            pass
            
        return self.expand_query_concepts(query)
    
    def _build_expansion_prompt(self, query: str) -> str:
        """Build prompt for LLM query expansion."""
        return (
            f"Expand this research query to include related measurable variables:\n"
            f"Query: \"{query}\"\n\n"
            "Generate 3-5 specific, measurable variables that could be used to study this concept.\n"
            "Focus on variables that would exist in government administrative datasets.\n"
            "Return only the variable terms, separated by spaces.\n\n"
            "Examples:\n"
            "- \"poverty status\" → \"household income welfare payments housing costs employment status\"\n"
            "- \"health outcomes\" → \"hospital admissions medical procedures mortality health services\"\n"
            "- \"education quality\" → \"graduation rates qualifications academic achievement\"\n\n"
            f"For \"{query}\" return:"
        )
    
    def _parse_llm_expansion(self, response: str) -> List[str]:
        """Parse LLM expansion response."""
        try:
            # Clean the response
            cleaned = response.strip().replace('\n', ' ')
            
            # Extract terms (simple space-separated approach)
            terms = [term.strip() for term in cleaned.split() if len(term.strip()) > 2]
            
            # Limit to reasonable number
            return terms[:5]
            
        except Exception:
            return []
    
    def generate_query_variants(self, query: str, topic: str) -> List[Tuple[str, str]]:
        """Generate multiple query variants for comprehensive search."""
        variants = []
        
        # 1. Original query (but preprocessed to remove noise words)
        preprocessed_original = self.preprocess_query(query)
        variants.append(("original", preprocessed_original))
        
        # 2. Concept-expanded query (from preprocessed)
        concept_expanded = self.expand_query_concepts(preprocessed_original)
        if concept_expanded != preprocessed_original:
            variants.append(("concept", concept_expanded))
        
        # 3. Domain-enhanced query (from preprocessed)
        domain_enhanced = self.enhance_query_by_domain(preprocessed_original, topic)
        if domain_enhanced != preprocessed_original:
            variants.append(("domain", domain_enhanced))
        
        # 4. LLM-expanded query (from preprocessed)
        if self.llm_client:
            llm_expanded = self.llm_expand_query(preprocessed_original)
            if llm_expanded != preprocessed_original:
                variants.append(("llm", llm_expanded))
        
        # 5. Add concept-specific variable searches (use original query for concept detection)
        query_lower = query.lower()
        for concept in self.concept_mappings:
            if concept in query_lower:
                suggested_vars = self.get_context_specific_variables(concept)
                for var in suggested_vars[:3]:  # Limit to top 3
                    # Preprocess concept-specific variables too to remove noise words
                    preprocessed_var = self.preprocess_query(var)
                    variants.append(("concept_specific", preprocessed_var))
        
        return variants
    
    def get_expansion_weights(self) -> Dict[str, float]:
        """Get weights for different expansion types."""
        return {
            'original': 1.0,
            'concept': 0.85,
            'domain': 0.9,
            'llm': 0.8,
            'concept_specific': 0.95
        }
    
    def detect_query_concepts(self, query: str) -> List[str]:
        """Detect which concepts are present in the query."""
        detected = []
        query_lower = query.lower()
        
        for concept in self.concept_mappings:
            if concept in query_lower:
                detected.append(concept)
        
        return detected
    
    def _initialize_noise_words(self) -> List[str]:
        """Initialize list of noise words that should be filtered from queries."""
        return [
            'status',  # Main culprit - matches everything with "status"
            'level',   # Too generic
            'type',    # Too generic  
            'kind',    # Too generic
            'form',    # Too generic
            'way',     # Too generic
            'manner',  # Too generic
            'method',  # Too generic
            'approach' # Too generic
        ]
    
    def preprocess_query(self, query: str) -> str:
        """Remove noise words from query to improve search relevance."""
        import re
        
        # Convert to lowercase for processing
        processed_query = query.lower()
        
        # Define legitimate demographic status terms that should keep "status"
        legitimate_status_queries = [
            'indigenous status', 'marital status', 'citizenship status', 
            'visa status', 'immigration status', 'migration status'
        ]
        
        # Define concept-based status terms where "status" is usually noise
        concept_noise_patterns = [
            'poverty status', 'health status', 'education status',
            'economic status', 'social status', 'financial status'
        ]
        
        # Define employment-specific handling
        employment_status_pattern = 'employment status'
        
        # Check query types
        is_legitimate_status = any(term in processed_query for term in legitimate_status_queries)
        is_concept_noise = any(pattern in processed_query for pattern in concept_noise_patterns)
        is_employment_status = employment_status_pattern in processed_query
        
        # Remove noise words with special handling for "status"
        for noise_word in self.noise_words:
            if noise_word == 'status':
                # Handle different types of status queries
                if is_employment_status:
                    # For employment status, replace with more specific terms
                    processed_query = processed_query.replace('employment status', 'employment labor force participation')
                elif is_concept_noise and not is_legitimate_status:
                    # Remove "status" from concept noise (poverty status -> poverty)
                    pattern = r'\b' + re.escape(noise_word) + r'\b'
                    processed_query = re.sub(pattern, '', processed_query)
                # For legitimate demographic status, keep "status" unchanged
            else:
                # Remove other noise words normally
                pattern = r'\b' + re.escape(noise_word) + r'\b'
                processed_query = re.sub(pattern, '', processed_query)
        
        # Clean up extra spaces
        processed_query = ' '.join(processed_query.split())
        
        # If query becomes empty after removing noise words, return original
        if not processed_query.strip():
            return query
            
        return processed_query
    
    def preprocess_and_expand(self, query: str, topic: str = None) -> str:
        """Preprocess query by removing noise words, then apply expansion."""
        # First remove noise words
        cleaned_query = self.preprocess_query(query)
        
        # Then apply concept expansion
        if topic:
            expanded = self.enhance_query_by_domain(cleaned_query, topic)
        else:
            expanded = self.expand_query_concepts(cleaned_query)
            
        return expanded