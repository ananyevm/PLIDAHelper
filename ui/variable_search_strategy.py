"""
Variable search strategy classes for handling different types of variables.
This module contains the logic for determining how to search for different variable types.
"""

import re
from abc import ABC, abstractmethod


class VariableType:
    """Identifies the type of a variable based on its description."""
    
    @staticmethod
    def get_type(var_desc):
        """
        Determine the type of variable based on its description.
        Returns the appropriate search strategy.
        """
        if VariableType.is_indigenous(var_desc):
            return 'indigenous'
        elif VariableType.is_core_demographic(var_desc):
            return 'core_demographic'
        elif VariableType.is_employment(var_desc):
            return 'employment'
        elif VariableType.is_location(var_desc):
            return 'location'
        else:
            return 'general'
    
    @staticmethod
    def is_indigenous(var_desc):
        """Check if variable is marked as Indigenous/Aboriginal related."""
        # Primary check: explicit (indig) suffix
        if var_desc.strip().endswith('(indig)'):
            return True
        
        # Fallback check: Indigenous-related terms (in case OpenAI doesn't add suffix)
        var_desc_lower = var_desc.lower()
        indigenous_terms = ['indigenous', 'aboriginal', 'torres strait', 'first nations', 'atsi']
        
        # Check if the variable description contains Indigenous terms
        if any(term in var_desc_lower for term in indigenous_terms):
            return True
            
        return False
    
    @staticmethod
    def is_core_demographic(var_desc):
        """Check if variable should only be searched in CORE datasets."""
        var_desc_lower = var_desc.lower()
        
        # Exact demographic terms (excluding Indigenous terms which have their own handler)
        exact_terms = ['sex', 'gender', 'ethnicity']
        if any(term in var_desc_lower for term in exact_terms):
            return True
            
        # Word boundary terms
        boundary_terms = ['age']
        for term in boundary_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', var_desc_lower):
                return True
        
        # Specific demographic phrases
        specific_phrases = [
            'year of birth', 'month of birth', 'date of birth', 'country of birth',
            'age of respondent', 'sex of respondent', 'gender of respondent'
        ]
        if any(phrase in var_desc_lower for phrase in specific_phrases):
            return True
        
        # Simple demographic requests
        if var_desc_lower.strip() in ['marital status', 'marital', 'marriage status', 'married']:
            return True
            
        if var_desc_lower.strip() in ['state', 'state of residence']:
            return True
            
        # Simple state-related variables
        if re.search(r'\bstate\b', var_desc_lower) and len(var_desc_lower.split()) <= 3:
            return True
            
        return False
    
    @staticmethod
    def is_employment(var_desc):
        """Check if variable is employment-related."""
        employment_keywords = [
            'employment status', 'employment', 'employed', 'unemployed', 
            'job status', 'work status', 'occupation', 'workforce',
            'labor force', 'labour force', 'jobseeker', 'employment type'
        ]
        var_desc_lower = var_desc.lower()
        return any(keyword in var_desc_lower for keyword in employment_keywords)
    
    @staticmethod
    def is_location(var_desc):
        """Check if variable is location/geographic related."""
        location_keywords = [
            'location', 'geographic', 'geography', 'area', 'region', 'suburb',
            'postcode', 'address', 'state', 'territory', 'city', 'town',
            'residential', 'residence', 'place', 'spatial', 'locality'
        ]
        var_desc_lower = var_desc.lower()
        return any(keyword in var_desc_lower for keyword in location_keywords)


class SearchStrategy(ABC):
    """Abstract base class for variable search strategies."""
    
    @abstractmethod
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Execute the search strategy for this variable type."""
        pass
    
    @abstractmethod
    def enhance_query(self, var_desc, base_query):
        """Enhance the search query for this variable type."""
        pass


class IndigenousSearchStrategy(SearchStrategy):
    """Search strategy for Indigenous/Aboriginal variables - uses COMBINED dataset only."""
    
    def enhance_query(self, var_desc, base_query):
        """Remove (indig) suffix for cleaner search."""
        if base_query.strip().endswith('(indig)'):
            return base_query.replace('(indig)', '').strip()
        return base_query
    
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Search only in COMBINED dataset for Indigenous variables."""
        from ui.display import ResultDisplay
        display = ResultDisplay()
        return display._search_combined_only_variable(search_query, search_engine)


class CoreDemographicSearchStrategy(SearchStrategy):
    """Search strategy for core demographic variables - uses CORE dataset only."""
    
    def enhance_query(self, var_desc, base_query):
        """No enhancement needed for core demographics."""
        return base_query
    
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Search only in CORE dataset for demographic variables."""
        from ui.display import ResultDisplay
        display = ResultDisplay()
        return display._search_core_only_variable(search_query, search_engine)


class EmploymentSearchStrategy(SearchStrategy):
    """Search strategy for employment variables - adds jobseeker context."""
    
    def enhance_query(self, var_desc, base_query):
        """Add jobseeker context to employment queries."""
        return base_query + " jobseeker"
    
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Standard search with employment enhancement."""
        return self._standard_search(var_desc, search_query, search_engine, **kwargs)
    
    def _standard_search(self, var_desc, search_query, search_engine, **kwargs):
        """Execute standard semantic search."""
        index_name = kwargs.get('index_name', 'plida4')
        relevant_datasets = kwargs.get('relevant_datasets')
        topic = kwargs.get('topic')
        
        results = search_engine.search_variables(
            search_query, 
            index_name, 
            boost_datasets=relevant_datasets,
            topic=topic,
            use_expansion=True,
            use_openai_relevance=True,
            conceptual_variable=var_desc
        )
        return search_engine.deduplicate_results(results)


class LocationSearchStrategy(SearchStrategy):
    """Search strategy for location variables - adds geographic context."""
    
    def enhance_query(self, var_desc, base_query):
        """Add geographic codes for location queries (except simple state queries)."""
        if not re.search(r'\bstate\b', var_desc.lower()):
            return base_query + " SA1 SA2 SA3 SA4"
        return base_query
    
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Standard search with location enhancement."""
        return EmploymentSearchStrategy()._standard_search(var_desc, search_query, search_engine, **kwargs)


class GeneralSearchStrategy(SearchStrategy):
    """Default search strategy for general variables."""
    
    def enhance_query(self, var_desc, base_query):
        """No enhancement for general variables."""
        return base_query
    
    def search(self, var_desc, search_query, search_engine, **kwargs):
        """Standard semantic search."""
        return EmploymentSearchStrategy()._standard_search(var_desc, search_query, search_engine, **kwargs)


class VariableSearchCoordinator:
    """Coordinates variable searches using appropriate strategies."""
    
    def __init__(self):
        self.strategies = {
            'indigenous': IndigenousSearchStrategy(),
            'core_demographic': CoreDemographicSearchStrategy(),
            'employment': EmploymentSearchStrategy(),
            'location': LocationSearchStrategy(),
            'general': GeneralSearchStrategy()
        }
    
    def search_variable(self, var_desc, search_engine, search_filters, **kwargs):
        """
        Search for a variable using the appropriate strategy.
        
        Args:
            var_desc: Variable description
            search_engine: Search engine instance
            search_filters: Search filters instance
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        # Determine variable type and strategy
        var_type = VariableType.get_type(var_desc)
        strategy = self.strategies[var_type]
        
        # Get base search parameters
        base_query = var_desc
        index_name, dataset_note, additional_searches = search_filters.get_appropriate_index(
            var_desc, kwargs.get('topic', '')
        )
        
        # Enhance query based on strategy
        enhanced_query = strategy.enhance_query(var_desc, base_query)
        
        # Execute search
        search_params = {
            'index_name': index_name,
            'relevant_datasets': kwargs.get('relevant_datasets'),
            'topic': kwargs.get('topic')
        }
        
        results = strategy.search(var_desc, enhanced_query, search_engine, **search_params)
        
        return results, dataset_note, additional_searches