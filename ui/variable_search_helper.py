"""
Helper class to simplify variable search operations.
This consolidates the variable search logic to make it more understandable.
"""

from .variable_search_strategy import VariableSearchCoordinator


class VariableSearchHelper:
    """Helper class that simplifies variable search operations."""
    
    def __init__(self):
        self.search_coordinator = VariableSearchCoordinator()
    
    def search_and_process_variable(self, var_desc, search_engine, search_filters, 
                                  query_analyzer=None, user_input=None, topic=None, 
                                  relevant_datasets=None, is_longitudinal=False, 
                                  available_datasets=None, variable_category=None):
        """
        Complete pipeline for searching and processing a single variable.
        
        Args:
            var_desc: Variable description
            search_engine: Search engine instance
            search_filters: Search filters instance
            query_analyzer: Query analyzer for population matching (optional)
            user_input: Original user input for context (optional)
            topic: Topic for search context (optional)
            relevant_datasets: List of datasets to boost (optional)
            is_longitudinal: Whether analysis is longitudinal (optional)
            available_datasets: Available datasets for recommendations (optional)
            variable_category: Variable category for ACLD substitution (optional)
            
        Returns:
            tuple: (search_results, dataset_ids, additional_searches)
        """
        # Step 1: Execute the search using appropriate strategy
        var_results, dataset_note, additional_searches = self.search_coordinator.search_variable(
            var_desc,
            search_engine,
            search_filters,
            topic=topic,
            relevant_datasets=relevant_datasets
        )
        
        # Step 2: Apply population filtering if available
        if user_input and query_analyzer and var_results:
            var_results = query_analyzer.check_population_match(var_desc, var_results, search_engine)
        
        # Step 2.5: Apply ACLD to CENSUS substitution if needed
        if var_results:
            from .display import ResultDisplay
            display = ResultDisplay()
            # Apply substitution with category if available, otherwise assume control for all ACLD variables
            category_to_use = variable_category or 'control'  # Default to control for all ACLD variables
            
            var_results = display.substitute_acld_variables_with_census(var_results, is_longitudinal, category_to_use)
        
        # Step 3: Process additional searches for employment variables
        additional_results = []
        for additional in additional_searches:
            enhanced_query = var_desc + additional['query_suffix']
            add_results = search_engine.search_variables(
                enhanced_query, 
                additional['index'], 
                boost_datasets=relevant_datasets,
                topic=topic,
                use_expansion=True,
                use_openai_relevance=True,
                conceptual_variable=enhanced_query
            )
            add_results = search_engine.deduplicate_results(add_results)
            
            # Apply population filtering to additional results
            if user_input and query_analyzer and add_results:
                add_results = query_analyzer.check_population_match(enhanced_query, add_results, search_engine)
            
            # Apply ACLD to CENSUS substitution to additional results
            if add_results:
                from .display import ResultDisplay
                display = ResultDisplay()
                # Apply substitution with category if available, otherwise assume control for all ACLD variables
                category_to_use = variable_category or 'control'  # Default to control for all ACLD variables
                
                add_results = display.substitute_acld_variables_with_census(add_results, is_longitudinal, category_to_use)
            
            if add_results:
                additional_results.append((add_results, additional['note']))
        
        return var_results, dataset_note, additional_results
    
    def extract_dataset_ids_from_results(self, search_results):
        """
        Extract dataset IDs from search results with filtering and deduplication.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            Set of dataset IDs
        """
        if not search_results:
            return set()
        
        # Import here to avoid circular imports
        from ui.display import ResultDisplay
        display = ResultDisplay()
        
        # Deduplicate search results based on dataset, variable name, and description
        search_results = display._deduplicate_variable_results(search_results)
        
        # Filter results by OpenAI relevance score (only show variables with >51% relevance)
        search_results = display._filter_by_relevance_score(search_results, min_threshold=0.51)
        
        # Collect all dataset IDs from results
        dataset_ids = set()
        for result in search_results:
            dataset_id = result['row'].get('dataset_id', '')
            if dataset_id:
                dataset_ids.add(dataset_id)
        
        return dataset_ids
    
    def collect_datasets_from_variables(self, variables, search_engine, search_filters, 
                                      topic=None, relevant_datasets=None, user_input=None, 
                                      query_analyzer=None):
        """
        Collect dataset IDs from a list of variables without displaying results.
        
        Args:
            variables: List of variable descriptions
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic (optional)
            relevant_datasets: List of dataset names to prioritize (optional)
            user_input: Original user input (optional)
            query_analyzer: Query analyzer instance (optional)
            
        Returns:
            List of dataset ID sets
        """
        all_dataset_ids = []
        
        for var_desc in variables:
            # Search for the variable
            var_results, _, additional_results = self.search_and_process_variable(
                var_desc, search_engine, search_filters, query_analyzer, 
                user_input, topic, relevant_datasets, is_longitudinal=False, 
                available_datasets=None, variable_category=None
            )
            
            # Collect dataset IDs from main results
            if var_results:
                dataset_ids = self.extract_dataset_ids_from_results(var_results)
                if dataset_ids:
                    all_dataset_ids.append(dataset_ids)
            
            # Collect dataset IDs from additional results
            for add_results, _ in additional_results:
                if add_results:
                    dataset_ids = self.extract_dataset_ids_from_results(add_results)
                    if dataset_ids:
                        all_dataset_ids.append(dataset_ids)
        
        return all_dataset_ids