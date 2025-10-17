import streamlit as st
import re
from .components import UIComponents
from .variable_search_helper import VariableSearchHelper
from utils import truncate_description
from config.topics import TOPIC_QUERIES
from embeddings import EmbeddingManager

class ResultDisplay:
    def __init__(self):
        self.ui = UIComponents()
        self.embedding_manager = EmbeddingManager()
        self.search_helper = VariableSearchHelper()
    
    def display_low_relevance(self):
        """Display message for low relevance queries."""
        container = st.empty()
        self.ui.stream_text(container, 
            "Your question may not be answerable with PLIDA or lacks specificity. Try again.")
    
    def display_dataset_results(self, user_input, dataset_results, topic, datasets_df, search_engine, query_analyzer=None):
        """Display relevant datasets."""
        st.subheader("Most Relevant Datasets")
        container = st.empty()
        self.ui.stream_text(container, f"**Research Question:** {user_input}")
        
        relevant_datasets = []
        
        # Display main results
        for result in dataset_results:
            row = result['row']
            score = result['score']
            
            # Skip ACLD for healthcare
            if not (topic == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                truncated_desc = truncate_description(row['dataset_description'], query_analyzer=query_analyzer, context_type="dataset")
                dataset_text = (
                    f"- **Dataset:** {row['dataset_id']} | "
                    f"**Name:** {row['dataset_name']} | "
                    f"**Description:** {truncated_desc}"
                )
                dataset_container = st.empty()
                self.ui.stream_text(dataset_container, dataset_text)
                relevant_datasets.append(row['dataset_name'])
        
        # Suggest DOMINO for specific topics
        if topic in ["unemployment", "social services"]:
            self._suggest_domino(datasets_df, relevant_datasets, topic, query_analyzer)
        
        # Topic-specific search
        self._display_topic_datasets(topic, datasets_df, search_engine, relevant_datasets)
        
        return relevant_datasets
    
    def _suggest_domino(self, datasets_df, relevant_datasets, topic, query_analyzer=None):
        """Suggest DOMINO dataset for relevant topics."""
        domino_row = datasets_df[datasets_df["dataset_id"].str.upper() == "DOMINO"]
        if not domino_row.empty:
            domino_row = domino_row.iloc[0]
            if domino_row['dataset_name'] not in relevant_datasets:
                truncated_desc = truncate_description(domino_row['dataset_description'], query_analyzer=query_analyzer, context_type="dataset")
                dataset_text = (
                    f"- **Dataset ID:** {domino_row['dataset_id']} | "
                    f"**Name:** {domino_row['dataset_name']} | "
                    f"**Description:** {truncated_desc}"
                )
                dataset_container = st.empty()
                self.ui.stream_text(dataset_container, dataset_text)
                relevant_datasets.append(domino_row['dataset_name'])
    
    def _display_topic_datasets(self, topic, datasets_df, search_engine, relevant_datasets):
        """Display topic-specific dataset suggestions."""
        topic_query = TOPIC_QUERIES.get(topic, "")
        if topic_query:
            topic_results = search_engine.search_datasets(topic_query)
            
            topic_container = st.empty()
            self.ui.stream_text(topic_container, f"{topic.capitalize()}-related datasets you might consider:")
            
            for result in topic_results:
                row = result['row']
                if not (topic == "healthcare" and row["dataset_id"].upper() == "ACLD") and \
                   row['dataset_name'] not in relevant_datasets:
                    relevant_datasets.append(row['dataset_name'])
            
            updated_list_container = st.empty()
            self.ui.stream_text(updated_list_container, 
                              "\n".join([f"- {name}" for name in relevant_datasets]))
    
    def substitute_acld_variables_with_census(self, search_results, is_longitudinal, variable_category=None):
        """Substitute ACLD variables with CENSUS equivalents when longitudinal analysis is not needed."""
        if is_longitudinal:
            return search_results  # Keep ACLD for longitudinal analysis
        
        substituted_results = []
        for result in search_results:
            row = result['row']
            if row.get('dataset_id', '').upper() == 'ACLD':
                # Always try to find the equivalent variable in CENSUS dataset
                census_variable = self._find_variable_in_census(
                    row.get('variable_name', ''), 
                    row.get('description', '')
                )
                
                if census_variable:
                    # Create CENSUS equivalent with actual CENSUS data
                    census_result = {
                        'score': result['score'],
                        'row': {
                            'dataset_id': 'CENSUS',
                            'dataset': census_variable['dataset'],
                            'variable_name': census_variable['variable_name'],
                            'description': census_variable['description']
                        }
                    }
                    # Copy other fields if they exist
                    for key in ['openai_relevance', 'relevance_explanation', 'population_match', 'is_core_alternative']:
                        if key in result:
                            census_result[key] = result[key]
                    substituted_results.append(census_result)
                else:
                    # If no exact match found, create generic CENSUS equivalent with ACLD variable info
                    # Get CENSUS dataset name from datasets.csv
                    csv_info = self._get_dataset_info_from_csv('CENSUS')
                    if csv_info:
                        census_dataset_name = csv_info['name']
                    else:
                        census_dataset_name = 'Census of Population and Housing 2011'
                    
                    census_result = {
                        'score': result['score'],
                        'row': {
                            'dataset_id': 'CENSUS',
                            'dataset': census_dataset_name,
                            'variable_name': row.get('variable_name', ''),
                            'description': row.get('description', '')
                        }
                    }
                    # Copy other fields if they exist
                    for key in ['openai_relevance', 'relevance_explanation', 'population_match', 'is_core_alternative']:
                        if key in result:
                            census_result[key] = result[key]
                    substituted_results.append(census_result)
            else:
                substituted_results.append(result)
        
        return substituted_results

    def _find_variable_in_census(self, variable_name, variable_description):
        """Find a variable in CENSUS dataset by matching variable name and description."""
        try:
            import pandas as pd
            
            # Load plida4.csv to search for CENSUS variables
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Filter for CENSUS dataset
            census_variables = plida4_df[plida4_df['dataset_id'] == 'CENSUS']
            
            if census_variables.empty:
                return None
            
            # Try exact variable name match first
            exact_match = census_variables[
                census_variables['variable_name'].str.upper() == variable_name.upper()
            ]
            
            if not exact_match.empty:
                # Return the first exact match
                match = exact_match.iloc[0]
                return {
                    'dataset': match['dataset'],
                    'variable_name': match['variable_name'],
                    'description': match['description']
                }
            
            # Conservative matching - only for very specific cases
            # Use exact variable name patterns that are known to exist in CENSUS
            variable_name_upper = variable_name.upper()
            variable_description_lower = variable_description.lower()
            
            # Map of common ACLD variable patterns to CENSUS equivalents
            known_mappings = {
                # Employment related
                'LFST': ['LFSF'],  # Labour Force Status
                'EMPS': ['EMPP', 'SIEMP'],  # Employment Status/Employees
                'INDUSTRY': ['INDP'],  # Industry
                'OCCUPATION': ['OCCP'],  # Occupation
                # Demographics
                'AGE': ['AGEP'],  # Age
                'SEX': ['SEXP'],  # Sex
            }
            
            # Try pattern-based matching for known cases
            for acld_pattern, census_patterns in known_mappings.items():
                if acld_pattern in variable_name_upper or acld_pattern.lower() in variable_description_lower:
                    for census_pattern in census_patterns:
                        # Look for exact census variable name match
                        census_match = census_variables[
                            census_variables['variable_name'].str.contains(census_pattern, case=False, na=False)
                        ]
                        
                        if not census_match.empty:
                            match = census_match.iloc[0]
                            return {
                                'dataset': match['dataset'],
                                'variable_name': match['variable_name'],
                                'description': match['description']
                            }
            
            # Very conservative description matching - only for clear cases
            if 'labour force status' in variable_description_lower:
                lfst_vars = census_variables[census_variables['description'].str.contains('Labour Force Status', case=False, na=False)]
                if not lfst_vars.empty:
                    match = lfst_vars.iloc[0]
                    return {
                        'dataset': match['dataset'],
                        'variable_name': match['variable_name'],
                        'description': match['description']
                    }
            
            if 'employment status' in variable_description_lower:
                emp_vars = census_variables[census_variables['description'].str.contains('Employment', case=False, na=False)]
                if not emp_vars.empty:
                    match = emp_vars.iloc[0]
                    return {
                        'dataset': match['dataset'],
                        'variable_name': match['variable_name'],
                        'description': match['description']
                    }
            
            if 'industry of employment' in variable_description_lower or 'industry' in variable_description_lower:
                ind_vars = census_variables[census_variables['description'].str.contains('Industry of Employment', case=False, na=False)]
                if not ind_vars.empty:
                    match = ind_vars.iloc[0]
                    return {
                        'dataset': match['dataset'],
                        'variable_name': match['variable_name'],
                        'description': match['description']
                    }
            
            return None
            
        except Exception as e:
            # If anything fails, return None to keep ACLD
            return None

    def _substitute_acld_datasets_in_results(self, dataset_results):
        """Substitute ACLD datasets with CENSUS in the results list."""
        substituted_results = []
        for result in dataset_results:
            row = result['row']
            if row.get('dataset_id', '').upper() == 'ACLD':
                # Create CENSUS equivalent
                census_result = {
                    'score': result['score'],
                    'row': {
                        'dataset_id': 'CENSUS',
                        'dataset_name': 'Census of Population and Housing 2011',
                        'dataset_description': 'Complete population coverage from the 2011 Census, providing comprehensive demographic and socio-economic information.'
                    }
                }
                substituted_results.append(census_result)
            else:
                substituted_results.append(result)
        
        return substituted_results

    def display_variable_results(self, variable_desc, index, search_results, dataset_note, query_analyzer=None, is_longitudinal=False, available_datasets=None, variable_category=None):
        """Display variable search results."""
        
        # Apply ACLD to CENSUS substitution if not longitudinal
        search_results = self.substitute_acld_variables_with_census(search_results, is_longitudinal, variable_category)
        
        # Deduplicate search results based on dataset, variable name, and description
        search_results = self._deduplicate_variable_results(search_results)
        
        # Filter results by OpenAI relevance score (only show variables with >51% relevance)
        search_results = self._filter_by_relevance_score(search_results, min_threshold=0.51)
        
        # Collect all dataset IDs from results for linking analysis
        dataset_ids = set()
        for result in search_results:
            dataset_id = result['row'].get('dataset_id', '')
            if dataset_id:
                dataset_ids.add(dataset_id)
        
        # Check if any variable has population coverage warning
        has_population_warning = self._check_for_population_warnings(search_results)
        
        # Separate regular results from CORE alternatives
        regular_results = [r for r in search_results if not r.get('is_core_alternative', False)]
        core_alternatives = [r for r in search_results if r.get('is_core_alternative', False)]
        
        # Display regular results first
        if regular_results:
            for result in regular_results:
                self._display_single_variable_result(result, query_analyzer=query_analyzer)
        
        # Display CORE alternatives if any
        if core_alternatives:
            for result in core_alternatives:
                self._display_single_variable_result(result, is_core_alternative=True, query_analyzer=query_analyzer)
        
        # If no variables met the relevance threshold, try to recommend datasets
        if not regular_results and not core_alternatives:
            if query_analyzer and available_datasets:
                # Get dataset recommendations from OpenAI
                recommendation = query_analyzer.recommend_datasets_for_conceptual_variable(
                    variable_desc, available_datasets
                )
                
                if recommendation['recommended_datasets']:
                    # Display dataset recommendations
                    st.markdown(f"We recommend referring to the documentation of these datasets for '{variable_desc}':")
                    
                    for dataset_id in recommendation['recommended_datasets']:
                        # Find the dataset info
                        dataset_info = next((ds for ds in available_datasets if ds.get('dataset_id') == dataset_id), None)
                        if dataset_info:
                            dataset_name = dataset_info.get('dataset_name', 'Unknown')
                            st.write(f"• **{dataset_id}**: {dataset_name}")
                    
                    # Add reasoning if provided
                    if recommendation['reasoning']:
                        st.write(f"{recommendation['reasoning']}")
                else:
                    # Fallback to original message
                    no_results_container = st.empty()
                    self.ui.stream_text(no_results_container, 
                        "We could not locate variables matching this description. However, the information we have access to is limited. "
                        "Please refer to the documentation of the datasets listed above—it's possible you'll find what you're looking for there.")
            else:
                # Fallback to original message when no query_analyzer or datasets available
                no_results_container = st.empty()
                self.ui.stream_text(no_results_container, 
                    "We could not locate variables matching this description. However, the information we have access to is limited. "
                    "Please refer to the documentation of the datasets listed above—it's possible you'll find what you're looking for there.")
        
        return dataset_ids, has_population_warning
    
    def _deduplicate_variable_results(self, search_results):
        """Remove duplicate variables with same dataset_id and variable_name."""
        seen = set()
        deduplicated = []
        
        for result in search_results:
            row = result['row']
            # Create a unique key based on dataset_id and variable name only
            # This catches duplicates across different years of the same dataset
            key = (
                row.get('dataset_id', '').strip(),
                row.get('variable_name', '').strip()
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)
        
        return deduplicated
    
    def _filter_by_relevance_score(self, search_results, min_threshold=0.51):
        """Filter search results to only include variables with OpenAI relevance score above threshold."""
        filtered_results = []
        
        for result in search_results:
            # Ensure the result has a relevance score (add default if missing)
            if 'openai_relevance' not in result:
                result['openai_relevance'] = 0.3
                result['relevance_explanation'] = "Relevance not evaluated - likely demographic variable with indirect relationship"
            
            # Only include results above the threshold
            relevance_score = result.get('openai_relevance', 0.0)
            if relevance_score > min_threshold:
                filtered_results.append(result)
        
        return filtered_results
    
    def _check_for_population_warnings(self, search_results):
        """Check if any of the search results has a population coverage warning."""
        for result in search_results:
            population_match = result.get('population_match')
            if population_match and population_match.get('match') == 'maybe':
                return True
        return False
    
    def _is_core_only_variable(self, var_desc):
        """Check if variable should only be searched in CORE datasets.
        
        This method should only return True for variables that are explicitly requesting
        basic demographic information that's primarily available in CORE datasets.
        """
        var_desc_lower = var_desc.lower()
        
        # Only trigger CORE-only search for very specific demographic requests
        # These should be variables that are explicitly asking for basic demographics
        
        # Exact demographic terms that clearly indicate CORE-only variables
        exact_demographic_terms = ['sex', 'gender', 'indigenous', 'aboriginal', 'torres strait', 'ethnicity']
        
        # Word boundary demographic terms with stricter context
        import re
        specific_demographic_terms = ['age']
        
        # Check exact demographic terms
        if any(term in var_desc_lower for term in exact_demographic_terms):
            return True
            
        # Check word boundary terms only for very specific contexts
        for term in specific_demographic_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', var_desc_lower):
                return True
        
        # Check specific demographic phrases - be more restrictive
        specific_demographic_phrases = [
            'year of birth', 'month of birth', 'date of birth', 'country of birth',
            'age of respondent', 'sex of respondent', 'gender of respondent'
        ]
        if any(phrase in var_desc_lower for phrase in specific_demographic_phrases):
            return True
        
        # Only trigger for explicit marital status requests, not incidental mentions
        if var_desc_lower.strip() in ['marital status', 'marital', 'marriage status', 'married']:
            return True
        
        # Only trigger for very basic state/location requests in demographic context
        # Exclude more complex location queries that should use expanded search
        if var_desc_lower.strip() in ['state', 'state of residence']:
            return True
            
        # Also check for any variable description that contains 'state' as a key concept
        # This catches cases like "state (demography)" or "state information"
        if re.search(r'\bstate\b', var_desc_lower) and len(var_desc_lower.split()) <= 3:
            # Simple state-related variable descriptions should use CORE
            return True
            
        return False
    
    def _search_core_only_variable(self, search_query, search_engine):
        """Search for variables only in CORE datasets by directly querying plida4.csv."""
        try:
            import pandas as pd
            
            # Preprocess query to remove noise words like 'status'
            if hasattr(search_engine, 'query_expansion'):
                preprocessed_query = search_engine.query_expansion.preprocess_query(search_query)
            else:
                # Fallback preprocessing if query_expansion not available
                import re
                preprocessed_query = search_query.lower()
                # Remove common noise words
                noise_words = ['status', 'level', 'type', 'kind']
                for noise_word in noise_words:
                    pattern = r'\b' + re.escape(noise_word) + r'\b'
                    preprocessed_query = re.sub(pattern, '', preprocessed_query)
                preprocessed_query = ' '.join(preprocessed_query.split())
            
            # Use preprocessed query for search
            if not preprocessed_query.strip():
                preprocessed_query = search_query  # Fallback to original if empty
                
            # Load plida4.csv directly
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Filter for CORE dataset only
            core_variables = plida4_df[plida4_df['dataset_id'] == 'CORE'].copy()
            
            if core_variables.empty:
                return []
            
            # Search for matches in CORE variables using preprocessed query
            results = []
            query_lower = preprocessed_query.lower()
            
            for _, row in core_variables.iterrows():
                description = str(row['description']).lower()
                variable_name = str(row['variable_name']).lower()
                combined_text = f"{description} {variable_name}"
                
                score = 0
                
                # Score based on query terms
                query_terms = query_lower.split()
                for term in query_terms:
                    if len(term) > 1:  # Allow 2+ character terms
                        # Exact variable name match gets highest priority
                        if term == variable_name:
                            score += 10  # Exact match gets top score
                        elif term in variable_name:
                            score += 5  # Partial variable name match
                        elif term in description:
                            score += 1  # Description match
                
                # Special handling for exact demographic matches
                if 'sex' in query_lower and variable_name == 'sex':
                    score += 15  # Boost exact sex match
                elif 'state' in query_lower and variable_name == 'state':
                    score += 15  # Boost exact state match
                elif 'gender' in query_lower and 'gender' in variable_name:
                    score += 15  # Boost gender matches
                
                # Enhanced state matching - catch various state-related queries
                if 'state' in query_lower:
                    if variable_name in ['state', 'state_asgs_2021', 'state_abbreviation']:
                        score += 10  # Strong boost for any state variable
                    elif 'state' in variable_name or 'state' in description:
                        score += 5   # Moderate boost for state-related variables
                
                # Special handling for age-related queries
                if 'age' in query_lower:
                    if any(age_term in combined_text for age_term in ['birth', 'born', 'dob', 'age']):
                        score += 2
                
                # Special handling for employment queries - boost employment matches, penalize irrelevant
                if 'employment' in search_query.lower():
                    if any(emp_term in combined_text for emp_term in ['employment', 'work', 'job', 'occupation', 'labor', 'labour']):
                        score += 5  # Boost employment-related variables
                    elif any(irrelevant_term in combined_text for irrelevant_term in ['marital', 'marriage', 'spouse']):
                        score -= 10  # Heavily penalize marital status for employment queries
                
                # General penalty for marital status when not specifically searching for it
                original_query_lower = search_query.lower()
                if 'marital' not in original_query_lower and 'marriage' not in original_query_lower:
                    if any(marital_term in combined_text for marital_term in ['marital', 'marriage', 'spouse']):
                        score -= 5  # Penalty for marital variables when not requested
                
                # Only include results with positive scores (after penalties)
                if score > 0:
                    # Normalize score (handle negative scores properly)
                    normalized_score = max(0.1, min(score / 10.0, 1.0))
                    
                    result = {
                        'score': normalized_score,
                        'row': {
                            'dataset_id': row['dataset_id'],
                            'dataset': row['dataset'],
                            'variable_name': row['variable_name'],
                            'description': row['description']
                        }
                    }
                    results.append(result)
            
            # Sort by score and deduplicate manually for CORE results
            # Use variable_name as secondary sort key for deterministic ordering in case of ties
            results = sorted(results, key=lambda x: (x['score'], x['row'].get('variable_name', '')), reverse=True)
            
            # Manual deduplication for CORE results (ignore dataset year differences)
            seen = set()
            deduplicated = []
            for result in results:
                row = result['row']
                # Only use variable name and description for deduplication, not dataset name
                key = (row.get('variable_name', '').strip(), row.get('description', '').strip())
                if key not in seen:
                    seen.add(key)
                    # Use the most recent dataset version (prefer higher score)
                    deduplicated.append(result)
            
            # Add OpenAI relevance scoring to CORE results
            final_results = deduplicated[:5]  # Get top 5 unique CORE results
            
            # Add OpenAI relevance scores if search_engine has relevance_scorer
            if hasattr(search_engine, 'relevance_scorer') and search_engine.relevance_scorer:
                try:
                    final_results = search_engine._add_openai_relevance_scores(final_results, search_query)
                except Exception as e:
                    # If OpenAI scoring fails, add default scores
                    for result in final_results:
                        result['openai_relevance'] = 0.5
                        result['relevance_explanation'] = "Could not evaluate relevance due to API error"
            
            return final_results
            
        except Exception as e:
            # Fallback: return empty list if search fails
            return []
    
    def _get_dataset_name_from_csv(self, dataset_id):
        """Get dataset name from datasets.csv by dataset_id."""
        try:
            import pandas as pd
            
            # Load datasets.csv
            datasets_df = pd.read_csv('resources/datasets.csv')
            
            # Find the dataset by ID
            dataset_row = datasets_df[datasets_df['dataset_id'] == dataset_id]
            
            if not dataset_row.empty:
                return dataset_row.iloc[0]['dataset_name']
            else:
                return None
                
        except Exception as e:
            return None

    def _get_dataset_info_from_csv(self, dataset_id):
        """Get dataset name and description from datasets.csv by dataset_id."""
        try:
            import pandas as pd
            
            # Load datasets.csv
            datasets_df = pd.read_csv('resources/datasets.csv')
            
            # Find the dataset by ID
            dataset_row = datasets_df[datasets_df['dataset_id'] == dataset_id]
            
            if not dataset_row.empty:
                row = dataset_row.iloc[0]
                return {
                    'name': row['dataset_name'],
                    'description': row['dataset_description']
                }
            else:
                return None
                
        except Exception as e:
            return None

    def _get_period_from_csv(self, dataset_id):
        """Get reference_period from periods.csv by dataset_id."""
        try:
            import pandas as pd
            
            # Load periods.csv
            periods_df = pd.read_csv('resources/periods.csv')
            
            # Find the dataset by ID
            period_row = periods_df[periods_df['dataset_id'] == dataset_id]
            
            if not period_row.empty:
                return period_row.iloc[0]['reference_period']
            else:
                return None
                
        except Exception as e:
            return None

    def _display_single_variable_result(self, result, is_core_alternative=False, query_analyzer=None):
        """Display a single variable result."""
        row = result['row']
        score = result['score']
        truncated_desc = truncate_description(row['description'], query_analyzer=query_analyzer, context_type="variable")
        
        # Removed prioritization indicators for cleaner display
        
        # Individual population warnings removed - now handled at section level
        
        # Get dataset name - use datasets.csv for non-CORE datasets, plida4.csv for CORE
        dataset_id = row.get('dataset_id', 'Unknown')
        if dataset_id.upper() == 'CORE':
            # For CORE dataset, use the dataset name from plida4.csv (row['dataset'])
            # Remove parenthetical information like "(2011 Census)"
            dataset_name = row['dataset']
            if '(' in dataset_name:
                dataset_display_name = dataset_name.split('(')[0].strip()
            else:
                dataset_display_name = dataset_name
        else:
            # For other datasets, look up the name from datasets.csv
            dataset_name_from_csv = self._get_dataset_name_from_csv(dataset_id)
            if dataset_name_from_csv:
                dataset_display_name = dataset_name_from_csv
            else:
                # Fallback to plida4.csv name if not found in datasets.csv
                dataset_display_name = row['dataset']
        
        # Get period information
        period_covered = self._get_period_from_csv(dataset_id)
        period_text = f" | **Period covered:** {period_covered}" if period_covered else ""
        
        match_text = (
            f"- **Dataset:** {dataset_id} - {dataset_display_name} | "
            f"**Variable:** {row['variable_name']} | "
            f"**Description:** {truncated_desc}{period_text}"
        )
        match_container = st.empty()
        self.ui.stream_text(match_container, match_text)
        
        # Population details removed - simple warning message is sufficient
        
        # Show ACLD notice if this is an ACLD variable
        if row.get('dataset_id') == 'ACLD':
            with st.expander(f"Census Data Alternative - {row['variable_name']}", expanded=False):
                st.write(
                    "**Alternative Available:** The full Census of Population and Housing should have the same variables. "
                    "Consider what you value more:\n"
                    "• **Cross-sectional representation** (full Census) - Complete population coverage\n"
                    "• **Linkage of individuals between Census years** (ACLD) - Longitudinal tracking with 5% sample"
                )
                st.caption("ACLD provides longitudinal linkage but covers only a 5% sample of the population, while the full Census provides complete population coverage for cross-sectional analysis.")
    
    def check_and_display_linking_strategy(self, dataset_ids_collection, is_longitudinal=False):
        """Check for SYNTHETIC_AEUID in datasets and display linking strategy."""
        import pandas as pd
        
        if not dataset_ids_collection:
            return
        
        # Flatten the collection of dataset ID sets into a single set
        all_dataset_ids = set()
        for dataset_ids in dataset_ids_collection:
            all_dataset_ids.update(dataset_ids)
        
        # Apply ACLD to CENSUS substitution if not longitudinal
        if not is_longitudinal:
            if 'ACLD' in all_dataset_ids:
                all_dataset_ids.remove('ACLD')
                all_dataset_ids.add('CENSUS')
        
        if not all_dataset_ids:
            return
            
        try:
            # Load plida4.csv to check for SYNTHETIC_AEUID
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Find datasets that contain SYNTHETIC_AEUID
            synthetic_datasets = plida4_df[
                (plida4_df['dataset_id'].isin(all_dataset_ids)) &
                (plida4_df['variable_name'] == 'SYNTHETIC_AEUID')
            ]['dataset_id'].unique().tolist()
            
            if synthetic_datasets:
                st.markdown("## Linkage")
                st.markdown("The real strength of PLIDA is the **capacity to link many data assets together**. Let's see which of the identified datasets could be linked.")
                
                st.write(
                    "The following datasets contain the **SYNTHETIC_AEUID** variable, which enables potential "
                    "data linking capabilities. You need to verify the specific linking rules and restrictions with the concordance files available in the ABS datalab."
                )
                
                # Remove duplicates from linkable datasets
                unique_synthetic_datasets = list(set(synthetic_datasets))
                
                # Display linkable datasets
                for dataset_id in sorted(unique_synthetic_datasets):
                    # Get dataset name from datasets.csv (except for CORE)
                    if dataset_id.upper() == 'CORE':
                        # For CORE, get from plida4 (more specific)
                        dataset_info = plida4_df[plida4_df['dataset_id'] == dataset_id].iloc[0]
                        dataset_name = dataset_info['dataset']
                    else:
                        # For other datasets, try to get from datasets.csv
                        csv_info = self._get_dataset_info_from_csv(dataset_id)
                        if csv_info:
                            dataset_name = csv_info['name']
                        else:
                            # Fallback to plida4
                            dataset_info = plida4_df[plida4_df['dataset_id'] == dataset_id].iloc[0]
                            dataset_name = dataset_info['dataset']
                    
                    st.write(f"• **{dataset_id}**: {dataset_name}")
                
                st.write(
                    "**Important:** While these datasets have linking potential through SYNTHETIC_AEUID, "
                    "you must verify the specific linking rules, restrictions, and documentation before "
                    "attempting to link datasets. Contact the PLIDA team for detailed linking guidelines."
                )
                
        except Exception as e:
            st.error(f"Error checking linking strategy: {e}")
    
    def display_debug_info(self, gpt_data):
        """Display debug information."""
        with st.expander("Raw GPT-4o Response (Debug)"):
            st.write(gpt_data.get('raw_response', ''))
        
        with st.expander("Parsed JSON Response"):
            st.json({k: v for k, v in gpt_data.items() if k != 'raw_response'})
        
        # Display medical condition detection debug info if available
        if gpt_data.get('medical_condition_detected'):
            with st.expander("Medical Condition Detection (Debug)"):
                medical_info = gpt_data['medical_condition_detected']
                st.json({k: v for k, v in medical_info.items() if k != 'raw_response'})
                if medical_info.get('raw_response'):
                    st.text("Raw Medical Detection Response:")
                    st.write(medical_info['raw_response'])
        
    
    def collect_variable_datasets(self, categorized, search_engine, search_filters, topic, relevant_datasets=None, user_input=None, query_analyzer=None, analysis_type="causal"):
        """Collect dataset IDs from variables without displaying them.
        
        Args:
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            relevant_datasets: List of selected dataset names to prioritize
            user_input: Original user input
            query_analyzer: Query analyzer instance
            analysis_type: Type of analysis ("causal" or "descriptive")
        
        Returns:
            List of dataset ID sets
        """
        all_dataset_ids = []
        
        if analysis_type == "causal":
            # Process causal variables
            for category in ['causal_variables', 'outcome_variables', 'control_variables']:
                if categorized.get(category):
                    dataset_ids = self._collect_variable_section_datasets(
                        categorized[category],
                        search_engine,
                        search_filters,
                        topic,
                        relevant_datasets,
                        user_input,
                        query_analyzer
                    )
                    if dataset_ids:
                        all_dataset_ids.extend(dataset_ids)
        else:
            # Process descriptive variables
            for category in ['main_variables', 'subgroups']:
                if categorized.get(category):
                    dataset_ids = self._collect_variable_section_datasets(
                        categorized[category],
                        search_engine,
                        search_filters,
                        topic,
                        relevant_datasets,
                        user_input,
                        query_analyzer
                    )
                    if dataset_ids:
                        all_dataset_ids.extend(dataset_ids)
        
        return all_dataset_ids
    
    def _collect_variable_section_datasets(self, variables, search_engine, search_filters, topic, relevant_datasets=None, user_input=None, query_analyzer=None):
        """Collect dataset IDs from a section of variables without displaying.
        
        Returns:
            List of dataset ID sets
        """
        all_dataset_ids = []
        
        # Use the helper to collect dataset IDs from all variables
        all_dataset_ids = self.search_helper.collect_datasets_from_variables(
            variables, search_engine, search_filters, topic, relevant_datasets, user_input, query_analyzer
        )
        
        return all_dataset_ids
    
    
    def display_datasets_and_variables(self, dataset_results, categorized, search_engine, search_filters, topic, ui, relevant_datasets, user_input, query_analyzer, causal_data, unique_dataset_ids, dataframes, loading_container=None, is_longitudinal=False):
        """Display datasets and variables together in a unified interface.
        
        Args:
            dataset_results: Initial dataset search results
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            relevant_datasets: List of selected dataset names
            user_input: Original user input
            query_analyzer: Query analyzer instance
            causal_data: Causal analysis data
            unique_dataset_ids: All unique dataset IDs found
            dataframes: DataFrame collection
            loading_container: Loading indicator container to clear when display starts
        """
        # First display all datasets found
        filtered_dataset_ids = []
        if dataset_results or unique_dataset_ids:
            filtered_dataset_ids = self._display_unified_datasets(dataset_results, unique_dataset_ids, dataframes, query_analyzer, topic, loading_container, is_longitudinal, user_input)
        
        # Then display variables
        variable_dataset_ids = []
        if categorized:
            if causal_data['is_causal'] and causal_data['confidence'] > 0.6:
                # Display causal variables
                variable_dataset_ids = self.display_causal_variables(
                    categorized,
                    search_engine,
                    search_filters,
                    topic,
                    ui,
                    relevant_datasets,
                    user_input,
                    query_analyzer,
                    is_longitudinal,
                    dataframes
                )
            else:
                # Display descriptive variables
                variable_dataset_ids = self.display_descriptive_variables(
                    categorized,
                    search_engine,
                    search_filters,
                    topic,
                    ui,
                    relevant_datasets,
                    user_input,
                    query_analyzer,
                    is_longitudinal,
                    dataframes
                )
        
        # Collect all dataset IDs for linking strategy (including Employment income datasets)
        all_datasets_for_linking = set(filtered_dataset_ids)
        if variable_dataset_ids:
            for dataset_collection in variable_dataset_ids:
                if isinstance(dataset_collection, set):
                    all_datasets_for_linking.update(dataset_collection)
                elif isinstance(dataset_collection, (list, tuple)):
                    all_datasets_for_linking.update(dataset_collection)
        
        
        # Display linking strategy at the end using all datasets
        if all_datasets_for_linking:
            self.check_and_display_linking_strategy([all_datasets_for_linking], is_longitudinal)
    
    def _display_unified_datasets(self, dataset_results, unique_dataset_ids, dataframes, query_analyzer, topic, loading_container=None, is_longitudinal=False, user_input=None):
        """Display unified list of datasets from both search results and variables.
        
        Args:
            dataset_results: Initial dataset search results
            unique_dataset_ids: All unique dataset IDs found
            dataframes: DataFrame collection
            query_analyzer: Query analyzer instance
            topic: Current topic
            loading_container: Loading indicator container to clear when display starts
            is_longitudinal: Whether the query requires longitudinal data
        """
        # Clear loading indicator immediately before displaying first content
        if loading_container is not None:
            loading_container.empty()
        
        # Note: ACLD to CENSUS substitution should already be done in main.py before this point
        
        # Create a comprehensive list of datasets
        dataset_info_list = []
        
        # Add datasets from initial search results
        if dataset_results:
            for result in dataset_results:
                row = result['row']
                dataset_id = row['dataset_id']
                # Skip ACLD for healthcare
                if not (topic == "healthcare" and dataset_id.upper() == "ACLD"):
                    # Get dataset info from datasets.csv (except for CORE)
                    if dataset_id.upper() == 'CORE':
                        # For CORE, use the info from dataframes (more specific)
                        dataset_name = row['dataset_name']
                        dataset_description = row['dataset_description']
                    else:
                        # For other datasets, try to get from datasets.csv
                        csv_info = self._get_dataset_info_from_csv(dataset_id)
                        if csv_info:
                            dataset_name = csv_info['name']
                            dataset_description = csv_info['description']
                        else:
                            # Fallback to original data
                            dataset_name = row['dataset_name']
                            dataset_description = row['dataset_description']
                    
                    from utils import truncate_description
                    truncated_desc = truncate_description(dataset_description, query_analyzer=query_analyzer, context_type="dataset")
                    dataset_info = {
                        'id': dataset_id,
                        'name': dataset_name,
                        'description': truncated_desc,
                        'source': 'search'
                    }
                    dataset_info_list.append(dataset_info)
        
        # Add datasets from variables that weren't already included
        included_ids = {info['id'] for info in dataset_info_list}
        for dataset_id in unique_dataset_ids:
            if dataset_id not in included_ids:
                # Apply ACLD to CENSUS substitution for variable-derived datasets
                if dataset_id.upper() == 'ACLD' and not is_longitudinal:
                    # Get CENSUS info from datasets.csv
                    csv_info = self._get_dataset_info_from_csv('CENSUS')
                    if csv_info:
                        dataset_name = csv_info['name']
                        dataset_description = csv_info['description']
                    else:
                        # Fallback to hardcoded values
                        dataset_name = 'Census of Population and Housing 2011'
                        dataset_description = 'Complete population coverage from the 2011 Census, providing comprehensive demographic and socio-economic information.'
                    
                    from utils import truncate_description
                    truncated_desc = truncate_description(dataset_description, query_analyzer=query_analyzer, context_type="dataset")
                    dataset_info = {
                        'id': 'CENSUS',
                        'name': dataset_name,
                        'description': truncated_desc,
                        'source': 'variables'
                    }
                    # Only add if CENSUS not already included
                    if 'CENSUS' not in included_ids:
                        dataset_info_list.append(dataset_info)
                        included_ids.add('CENSUS')
                else:
                    # Skip ACLD for healthcare
                    if not (topic == "healthcare" and dataset_id.upper() == "ACLD"):
                        # Get dataset info from datasets.csv (except for CORE)
                        if dataset_id.upper() == 'CORE':
                            # For CORE, look up from dataframes (more specific)
                            dataset_row = dataframes['datasets'][dataframes['datasets']['dataset_id'] == dataset_id]
                            if not dataset_row.empty:
                                row = dataset_row.iloc[0]
                                dataset_name = row['dataset_name']
                                dataset_description = row['dataset_description']
                            else:
                                continue  # Skip if not found
                        else:
                            # For other datasets, try to get from datasets.csv
                            csv_info = self._get_dataset_info_from_csv(dataset_id)
                            if csv_info:
                                dataset_name = csv_info['name']
                                dataset_description = csv_info['description']
                            else:
                                # Fallback to dataframes
                                dataset_row = dataframes['datasets'][dataframes['datasets']['dataset_id'] == dataset_id]
                                if not dataset_row.empty:
                                    row = dataset_row.iloc[0]
                                    dataset_name = row['dataset_name']
                                    dataset_description = row['dataset_description']
                                else:
                                    continue  # Skip if not found
                        
                        from utils import truncate_description
                        truncated_desc = truncate_description(dataset_description, query_analyzer=query_analyzer, context_type="dataset")
                        dataset_info = {
                            'id': dataset_id,
                            'name': dataset_name,
                            'description': truncated_desc,
                            'source': 'variables'
                        }
                        dataset_info_list.append(dataset_info)
        
        # Remove duplicates based on dataset_id
        seen_ids = set()
        deduplicated_list = []
        for dataset_info in dataset_info_list:
            if dataset_info['id'] not in seen_ids:
                seen_ids.add(dataset_info['id'])
                deduplicated_list.append(dataset_info)
        
        dataset_info_list = deduplicated_list
        
        # Display the unified dataset list with relevance scores
        if not dataset_info_list:
            return []  # Return empty list if no datasets to display
        
        if dataset_info_list:
            st.markdown("---")
            st.markdown("Here are the PLIDA data assets that might be relevant for your question:")
            
            # Add relevance scores if query_analyzer and user_input are available
            if query_analyzer and user_input:
                # Show progress indicator while scoring
                with st.spinner("Scoring dataset relevance..."):
                    # Score datasets for relevance
                    for dataset_info in dataset_info_list:
                        try:
                            score = query_analyzer.score_dataset_relevance(
                                user_input,
                                dataset_info['id'],
                                dataset_info['name'], 
                                dataset_info['description']
                            )
                            dataset_info['relevance_score'] = score
                        except Exception:
                            dataset_info['relevance_score'] = 0.5  # Default score if scoring fails
                
                # Filter out datasets with relevance < 0.1
                dataset_info_list = [dataset for dataset in dataset_info_list 
                                   if dataset.get('relevance_score', 0.5) >= 0.1]
                
                # Sort by relevance score (highest first)
                dataset_info_list.sort(key=lambda x: x.get('relevance_score', 0.5), reverse=True)
            
            # Display datasets without showing relevance scores
            for i, dataset_info in enumerate(dataset_info_list, 1):
                dataset_text = f"**{dataset_info['id']} - {dataset_info['name']}**: {dataset_info['description']}"
                st.markdown(f"{i}. {dataset_text}")
        
        # Return the filtered dataset IDs for linkage strategy
        return [dataset_info['id'] for dataset_info in dataset_info_list]

    def display_causal_variables(self, categorized, search_engine, search_filters, topic, ui, relevant_datasets=None, user_input=None, query_analyzer=None, is_longitudinal=False, dataframes=None):
        """Display variables categorized for causal analysis.
        
        Args:
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            relevant_datasets: List of selected dataset names to prioritize
        """
        # Show dataset context if relevant datasets exist
        all_dataset_ids = []
        current_number = 1
        
        # Create available_datasets list from dataframes for dataset recommendations
        available_datasets = []
        if dataframes and 'datasets' in dataframes:
            for _, row in dataframes['datasets'].iterrows():
                available_datasets.append({
                    'dataset_id': row.get('dataset_id', ''),
                    'dataset_name': row.get('dataset_name', ''),
                    'dataset_description': row.get('dataset_description', '')
                })
        
        # Start with narrative introduction
        st.markdown("Now let's try to identify specific variables in the PLIDA datasets which you might find useful for your causal analysis.")
        
        # Display Potential Causal Variables
        if categorized['causal_variables']:
            st.markdown("## Treatment Variables")
            st.markdown("We will start with potential treatment variables. These are variables that might cause or influence the outcome you're interested in:")
            dataset_ids, current_number = self._display_variable_section(
                categorized['causal_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "causal",
                relevant_datasets,
                user_input,
                query_analyzer,
                current_number,
                is_longitudinal,
                available_datasets
            )
            if dataset_ids:
                all_dataset_ids.extend(dataset_ids)
        
        # Display Outcome Variables
        if categorized['outcome_variables']:
            st.markdown("## Outcome Variables")
            st.markdown("Next, let's look at outcome variables. These are the variables being affected or measured as results:")
            dataset_ids, current_number = self._display_variable_section(
                categorized['outcome_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "outcome",
                relevant_datasets,
                user_input,
                query_analyzer,
                current_number,
                is_longitudinal,
                available_datasets
            )
            if dataset_ids:
                all_dataset_ids.extend(dataset_ids)
        
        # Display Control Variables
        if categorized['control_variables']:
            st.markdown("## Control Variables")
            st.markdown("Finally, we should consider control variables. These are variables that need to be controlled for valid causal inference:")
            dataset_ids, current_number = self._display_variable_section(
                categorized['control_variables'], 
                search_engine, 
                search_filters, 
                topic, 
                ui,
                "control",
                relevant_datasets,
                user_input,
                query_analyzer,
                current_number,
                is_longitudinal,
                available_datasets
            )
            if dataset_ids:
                all_dataset_ids.extend(dataset_ids)
        
        return all_dataset_ids
    
    def _display_variable_section(self, variables, search_engine, search_filters, topic, ui, category, relevant_datasets=None, user_input=None, query_analyzer=None, start_number=1, is_longitudinal=False, available_datasets=None):
        """Display a section of categorized variables.
        
        Args:
            variables: List of variable descriptions
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            category: Variable category (causal, outcome, control, etc.)
            relevant_datasets: List of selected dataset names to prioritize
        """
        # Collect dataset IDs for linking strategy
        all_dataset_ids = []
        employment_detected = False
        section_has_population_warning = False
        
        for i, var_desc in enumerate(variables):
            with st.container():
                # Display variable description
                container = st.empty()
                ui.stream_text(container, f"**{start_number + i}. {var_desc}**")
                
                # Check if this is an employment status variable
                if self._is_employment_status_variable(var_desc):
                    employment_detected = True
                
                # Search and process the variable using the helper
                var_results, dataset_note, additional_results = self.search_helper.search_and_process_variable(
                    var_desc, search_engine, search_filters, query_analyzer, 
                    user_input, topic, relevant_datasets, is_longitudinal, available_datasets, category
                )
                
                # Display main variable results
                if var_results:
                    dataset_ids, has_warning = self.display_variable_results(
                        var_desc, i, var_results, dataset_note, query_analyzer, is_longitudinal, available_datasets, category
                    )
                    if dataset_ids:
                        all_dataset_ids.append(dataset_ids)
                    if has_warning:
                        section_has_population_warning = True
                
                # Display additional search results
                for add_results, note in additional_results:
                    dataset_ids, has_warning = self.display_variable_results(
                        var_desc, i, add_results, note, query_analyzer, is_longitudinal, available_datasets, category
                    )
                    if dataset_ids:
                        all_dataset_ids.append(dataset_ids)
                    if has_warning:
                        section_has_population_warning = True
                
                # Removed horizontal line for cleaner display
        
        # Display PIT_ITR earnings variables if employment status detected
        if employment_detected:
            earnings_count, earnings_dataset_ids = self._display_pit_itr_earnings(search_engine, ui, relevant_datasets, start_number + len(variables))
            next_number = start_number + len(variables) + earnings_count
            # Add employment income dataset IDs to the collection
            if earnings_dataset_ids:
                all_dataset_ids.append(earnings_dataset_ids)
        else:
            next_number = start_number + len(variables)
        
        # Add section-level population coverage note if any variable requires verification
        if section_has_population_warning:
            st.write("**Note:** Please verify that population covered by these variables matches your research needs.")
        
        # Linking strategy will be displayed once at the end of the entire response
        return all_dataset_ids, next_number
    
    def display_descriptive_variables(self, categorized, search_engine, search_filters, topic, ui, relevant_datasets=None, user_input=None, query_analyzer=None, is_longitudinal=False, dataframes=None):
        """Display variables categorized for descriptive analysis.
        
        Args:
            categorized: Categorized variables
            search_engine: Search engine instance
            search_filters: Search filters instance
            topic: Current topic
            ui: UI components instance
            relevant_datasets: List of selected dataset names to prioritize
        """
        # Show dataset context if relevant datasets exist
        all_dataset_ids = []
        current_number = 1
        
        # Create available_datasets list from dataframes for dataset recommendations
        available_datasets = []
        if dataframes and 'datasets' in dataframes:
            for _, row in dataframes['datasets'].iterrows():
                available_datasets.append({
                    'dataset_id': row.get('dataset_id', ''),
                    'dataset_name': row.get('dataset_name', ''),
                    'dataset_description': row.get('dataset_description', '')
                })
        
        # Start with narrative introduction
        st.markdown("Now let's try to identify specific variables in the PLIDA datasets which you might find useful for your descriptive analysis.")
        
        # Display Main Variables
        if categorized['main_variables']:
            st.markdown("## Main Variables")
            st.markdown("We'll start with the main variables. These are the primary variables that directly address your research question:")
            dataset_ids, current_number = self._display_variable_section(
                categorized['main_variables'],
                search_engine,
                search_filters,
                topic,
                ui,
                "main",
                relevant_datasets,
                user_input,
                query_analyzer,
                current_number,
                is_longitudinal,
                available_datasets
            )
            if dataset_ids:
                all_dataset_ids.extend(dataset_ids)
        
        # Display Subgroups
        if categorized['subgroups']:
            st.markdown("## Subgroup Variables")
            st.markdown("Next, let's consider subgroup variables. These are useful for breaking down or segmenting your analysis:")
            dataset_ids, current_number = self._display_variable_section(
                categorized['subgroups'],
                search_engine,
                search_filters,
                topic,
                ui,
                "subgroup",
                relevant_datasets,
                user_input,
                query_analyzer,
                current_number,
                is_longitudinal,
                available_datasets
            )
            if dataset_ids:
                all_dataset_ids.extend(dataset_ids)
        
        return all_dataset_ids
    
    def display_medical_condition_suggestion(self, medical_info, datasets_df, relevant_datasets, query_analyzer=None):
        """Display PBS dataset suggestion for medical condition queries."""
        # Medical condition detection - datasets are automatically added to main list
        # No UI display needed here since PBS/MBS appear in numbered list
        
        return relevant_datasets
    
    
    def display_chat_history(self, chat_history):
        """Display the conversation history."""
        if not chat_history:
            return
        
        st.markdown("## **Chat History**")
        
        for i, entry in enumerate(chat_history):
            with st.expander(f"Query {i+1}: {entry['query'][:50]}{'...' if len(entry['query']) > 50 else ''}", expanded=False):
                st.markdown(f"**Query:** {entry['query']}")
                
                if entry['type'] == 'followup':
                    st.caption(f"**Combined with previous:** {entry.get('combined_query', '')}")
                    if entry.get('relevance'):
                        st.caption(f"**Relevance:** {entry['relevance']['reasoning']}")
                
                if entry.get('timestamp'):
                    import datetime
                    timestamp = datetime.datetime.fromtimestamp(entry['timestamp'])
                    st.caption(f"**Time:** {timestamp.strftime('%H:%M:%S')}")
        
        st.markdown("---")
    
    def display_stored_results(self, stored_results):
        """Display stored results from session state."""
        if not stored_results:
            return
        
        user_input = stored_results.get('user_input', '')
        gpt_data = stored_results.get('gpt_data', {})
        causal_data = stored_results.get('causal_data')
        relevant_datasets = stored_results.get('relevant_datasets', [])
        start_time = stored_results.get('start_time', 0)
        end_time = stored_results.get('end_time', 0)
        
        # Display query
        st.markdown(f"**Research Question:** {user_input}")
        
        # Check relevance
        if gpt_data.get('relevance_score', 0) < 6:
            self.display_low_relevance()
            return
        
        # Analysis type is now integrated into the question display
        
        # Medical and geographic analysis detected (info stored but not displayed procedurally)
        
        # Display relevant datasets
        if relevant_datasets:
            st.subheader("Relevant Datasets")
            for dataset in relevant_datasets:
                st.write(f"- {dataset}")
        
        # Display execution time
        if start_time and end_time:
            execution_time = end_time - start_time
            st.caption(f"Execution time: {execution_time:.2f} seconds")
        
        # Note about full results
        st.caption("*This is a summary of the latest results. The full analysis with variables was displayed when the query was processed.*")
    
    def _is_employment_status_variable(self, var_desc):
        """Check if variable description indicates employment status."""
        from .variable_search_strategy import VariableType
        return VariableType.is_employment(var_desc)
    
    def _search_combined_only_variable(self, search_query, search_engine):
        """Search for variables only in COMBINED dataset for Indigenous variables."""
        try:
            import pandas as pd
            
            # Remove (indig) suffix from search query for cleaner search
            if search_query.strip().endswith('(indig)'):
                search_query = search_query.replace('(indig)', '').strip()
            
            # Load plida4.csv directly
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Filter for COMBINED dataset only
            combined_variables = plida4_df[plida4_df['dataset_id'] == 'COMBINED'].copy()
            
            if combined_variables.empty:
                return []
            
            # Search for matches in COMBINED variables
            results = []
            query_lower = search_query.lower()
            
            for _, row in combined_variables.iterrows():
                description = str(row['description']).lower()
                variable_name = str(row['variable_name']).lower()
                combined_text = f"{description} {variable_name}"
                
                score = 0
                
                # Score based on query terms
                query_terms = query_lower.split()
                for term in query_terms:
                    if len(term) > 1:  # Allow 2+ character terms
                        # Exact variable name match gets highest priority
                        if term == variable_name:
                            score += 10  # Exact match gets top score
                        elif term in variable_name:
                            score += 5  # Partial variable name match
                        elif term in description:
                            score += 1  # Description match
                
                # Special handling for Indigenous-related keywords
                indigenous_keywords = ['indigenous', 'aboriginal', 'torres', 'strait', 'islander', 'first', 'nations', 'atsi']
                for keyword in indigenous_keywords:
                    if keyword in query_lower:
                        if keyword in combined_text:
                            score += 8  # High boost for Indigenous keyword matches
                
                # Only include results with positive scores
                if score > 0:
                    # Normalize score
                    normalized_score = max(0.1, min(score / 10.0, 1.0))
                    
                    result = {
                        'score': normalized_score,
                        'row': {
                            'dataset_id': row['dataset_id'],
                            'dataset': row['dataset'],
                            'variable_name': row['variable_name'],
                            'description': row['description']
                        }
                    }
                    results.append(result)
            
            # Sort by score and return top results
            results = sorted(results, key=lambda x: (x['score'], x['row'].get('variable_name', '')), reverse=True)
            
            # Add OpenAI relevance scoring if available
            final_results = results[:5]  # Get top 5 COMBINED results
            
            # Add OpenAI relevance scores if search_engine has relevance_scorer
            if hasattr(search_engine, 'relevance_scorer') and search_engine.relevance_scorer:
                try:
                    final_results = search_engine._add_openai_relevance_scores(final_results, search_query)
                except Exception as e:
                    # If OpenAI scoring fails, add default scores
                    for result in final_results:
                        result['openai_relevance'] = 0.7  # Higher default for Indigenous variables
                        result['relevance_explanation'] = "Indigenous variable from COMBINED dataset"
            
            return final_results
            
        except Exception as e:
            # Fallback: return empty list if search fails
            return []
    
    def _display_pit_itr_earnings(self, search_engine, ui, relevant_datasets, start_number):
        """Display PIT_ITR earnings variables when employment status is detected."""
        
        # Search for earnings variables in PIT_ITR
        earnings_results = search_engine.search_variables(
            "wages and salary", 
            "ato", 
            boost_datasets=relevant_datasets,
            use_openai_relevance=True,
            conceptual_variable="wages and salary"
        )
        
        displayed_count = 0
        dataset_ids = set()
        
        if earnings_results:
            # Filter to only show PIT_ITR results
            pit_itr_results = [result for result in earnings_results if result['row'].get('dataset_id', '').upper() == 'PIT_ITR']
            
            if pit_itr_results:
                # Remove duplicates based on variable_name
                seen_variables = set()
                unique_results = []
                for result in pit_itr_results:
                    var_name = result['row'].get('variable_name', 'Unknown')
                    if var_name not in seen_variables:
                        seen_variables.add(var_name)
                        unique_results.append(result)
                        # Collect dataset ID
                        dataset_id = result['row'].get('dataset_id', 'PIT_ITR')
                        dataset_ids.add(dataset_id)
                
                if unique_results:
                    # Display conceptual variable heading
                    conceptual_container = st.empty()
                    ui.stream_text(conceptual_container, f"**{start_number}. Employment income**")
                    displayed_count = 1
                    
                    # Display actual variables as sub-items
                    for i, result in enumerate(unique_results[:5]):
                        row = result['row']
                        
                        # Get dataset name using the same logic as other variables
                        dataset_id = row.get('dataset_id', 'PIT_ITR')
                        if dataset_id.upper() == 'CORE':
                            # Remove parenthetical information like "(2011 Census)"
                            dataset_name = row.get('dataset', 'Unknown')
                            if '(' in dataset_name:
                                dataset_display_name = dataset_name.split('(')[0].strip()
                            else:
                                dataset_display_name = dataset_name
                        else:
                            dataset_name_from_csv = self._get_dataset_name_from_csv(dataset_id)
                            if dataset_name_from_csv:
                                dataset_display_name = dataset_name_from_csv
                            else:
                                dataset_display_name = row.get('dataset', 'Unknown')
                        
                        # Get period information
                        period_covered = self._get_period_from_csv(dataset_id)
                        period_text = f" | **Period covered:** {period_covered}" if period_covered else ""
                        
                        # Create earnings variable display
                        earnings_text = (
                            f"**Dataset:** {dataset_id} - {dataset_display_name} | "
                            f"**Variable:** {row.get('variable_name', 'Unknown')} | "
                            f"**Description:** {row.get('description', 'No description')}{period_text}"
                        )
                        
                        container = st.empty()
                        ui.stream_text(container, f"   - {earnings_text}")
                    
                    # Add employment indicator note after the variables
                    st.write(
                        "**Note:** Non-zero values in these earnings variables indicate employment. "
                        "These tax return variables provide comprehensive employment income information that can be used "
                        "to identify employed individuals and measure employment earnings."
                    )
        
        return displayed_count, dataset_ids