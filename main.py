import time
import streamlit as st
from config import DATASETS_PATH, VARIABLES_PATH
from data import DataLoader
from search import IndexBuilder, SemanticSearchEngine, SearchFilters
from llm import QueryAnalyzer, CausalAnalyzer
from ui import UIComponents, ResultDisplay
from core import Response
# EmbeddingManager is initialized within other components

# Initialize session state for responses
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'structured_response' not in st.session_state:
    st.session_state.structured_response = None

# ================================
# DEBUGGING FUNCTIONS
# ================================
# The following functions provide access to structured query results for debugging:
# 
# 1. get_response_json(include_metadata=False) -> str
#    Returns JSON string of the latest response
# 
# 2. get_response_dict(include_metadata=False) -> dict  
#    Returns dictionary of the latest response
#
# 3. print_response_summary() -> None
#    Prints a formatted summary to console
#
# The structured response contains:
# - query_type: "causal" or "descriptive"
# - conceptual_variables: Each with variable_type:
#   * causal-treatment, causal-outcome, causal-control
#   * descriptive-main, descriptive-subgroup
# - actual_variables: List of (dataset_id, variable_name) matches
# - recommended_datasets: When no specific variables found
# 
# AUTOMATIC FEATURES:
# - Employment Income: If any conceptual variable is employment-related,
#   "Employment income" is automatically added as a conceptual variable
#   with PIT_ITR earnings variables (wages, salary, etc.)
# - Relevance Sorting: All actual variables within each conceptual variable
#   are automatically sorted by relevance score in descending order
# ================================

def get_response_json(include_metadata=False):
    """
    Get the JSON representation of the latest structured response.
    
    This function provides access to the internal structured representation of query results,
    including conceptual variables with their types (causal-treatment, causal-outcome, 
    causal-control, descriptive-main, descriptive-subgroup) and their associated actual 
    variables from datasets.
    
    Args:
        include_metadata: Whether to include internal metadata (confidence, topic, etc.)
        
    Returns:
        str: JSON string of the response, or None if no response available
        
    Example:
        >>> json_response = get_response_json()
        >>> # Returns structured JSON with query type, conceptual variables, and matches
    """
    if st.session_state.structured_response:
        return st.session_state.structured_response.to_json(include_metadata=include_metadata)
    return None

def get_response_dict(include_metadata=False):
    """
    Get the dictionary representation of the latest structured response.
    
    Args:
        include_metadata: Whether to include internal metadata
        
    Returns:
        dict: Dictionary of the response, or None if no response available
    """
    if st.session_state.structured_response:
        return st.session_state.structured_response.to_dict(include_metadata=include_metadata)
    return None

def print_response_summary():
    """
    Print a summary of the latest structured response to console.
    Useful for debugging in Python console or notebooks.
    """
    if not st.session_state.structured_response:
        print("No structured response available")
        return
    
    response = st.session_state.structured_response
    print(f"Query: {response.query}")
    print(f"Type: {response.query_type}")
    print(f"Conceptual Variables: {len(response.conceptual_variables)}")
    print(f"Relevant Datasets: {response.relevant_datasets}")
    print()
    
    for i, var in enumerate(response.conceptual_variables, 1):
        print(f"{i}. {var.name} ({var.variable_type}):")
        if var.actual_variables:
            # Sort variables by relevance score in descending order for display
            sorted_variables = sorted(
                var.actual_variables,
                key=lambda x: x.relevance_score or 0,
                reverse=True
            )
            for match in sorted_variables:
                relevance = f" [{match.relevance_score:.2f}]" if match.relevance_score else ""
                print(f"   • {match.dataset_id}.{match.variable_name}{relevance}")
        if var.recommended_datasets:
            print(f"   Recommended datasets: {', '.join(var.recommended_datasets)}")
        if not var.actual_variables and not var.recommended_datasets:
            print("   No matches or recommendations")
        print()

# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize all system components."""
    # Load data
    data_loader = DataLoader(DATASETS_PATH, VARIABLES_PATH)
    
    # Build indices
    index_builder = IndexBuilder(data_loader)
    indices, dataframes, embeddings = index_builder.build_all_indices()
    
    # Initialize query analyzer for LLM client
    query_analyzer = QueryAnalyzer()
    
    # Create search engine with LLM client for query expansion
    search_engine = SemanticSearchEngine(indices, dataframes, query_analyzer.llm_client)
    
    return data_loader, search_engine, dataframes, query_analyzer

def populate_response_with_variables(structured_response, categorized, search_engine, search_filters, query_analyzer, user_input, topic, relevant_datasets, is_longitudinal, dataframes):
    """Populate the Response object with actual variable search results."""
    if not structured_response or not categorized:
        return
    
    # Create available_datasets list from dataframes for dataset recommendations
    available_datasets = []
    if dataframes and 'datasets' in dataframes:
        for _, row in dataframes['datasets'].iterrows():
            available_datasets.append({
                'dataset_id': row.get('dataset_id', ''),
                'dataset_name': row.get('dataset_name', ''),
                'dataset_description': row.get('dataset_description', '')
            })
    
    # Import helper for variable search
    from ui.variable_search_helper import VariableSearchHelper
    from ui.variable_search_strategy import VariableType
    search_helper = VariableSearchHelper()
    
    # Check if any variable triggers employment income addition
    employment_detected = False
    
    # Process each conceptual variable
    for conceptual_var in structured_response.conceptual_variables:
        var_desc = conceptual_var.name
        
        # Check if this is an employment status variable
        if VariableType.is_employment(var_desc):
            employment_detected = True
        
        # Determine variable category for search
        if conceptual_var.variable_type.startswith('causal-'):
            category = conceptual_var.variable_type.split('-')[1]  # treatment, outcome, control
        else:
            category = conceptual_var.variable_type.split('-')[1]  # main, subgroup
        
        try:
            # Search for variables
            var_results, dataset_note, additional_results = search_helper.search_and_process_variable(
                var_desc, search_engine, search_filters, query_analyzer, 
                user_input, topic, relevant_datasets, is_longitudinal, available_datasets, category
            )
            
            # Apply ACLD to CENSUS substitution if not longitudinal
            from ui.display import ResultDisplay
            display_helper = ResultDisplay()
            var_results = display_helper.substitute_acld_variables_with_census(var_results, is_longitudinal, category)
            
            # Filter results by relevance score (only show variables with >51% relevance)
            filtered_results = display_helper._filter_by_relevance_score(var_results, min_threshold=0.51)
            
            # Add variable matches to Response object
            for result in filtered_results:
                row = result['row']
                structured_response.add_variable_match(
                    conceptual_var.name,
                    row.get('dataset_id', ''),
                    row.get('variable_name', ''),
                    row.get('description', ''),
                    result.get('openai_relevance', None)
                )
            
            # If no variables found, get dataset recommendations
            if not filtered_results and query_analyzer and available_datasets:
                recommendation = query_analyzer.recommend_datasets_for_conceptual_variable(
                    var_desc, available_datasets
                )
                if recommendation['recommended_datasets']:
                    structured_response.add_dataset_recommendations(
                        conceptual_var.name, 
                        recommendation['recommended_datasets']
                    )
                    
        except Exception as e:
            # If variable search fails, continue with next variable
            continue
    
    # Add employment income as conceptual variable if employment status detected
    if employment_detected:
        # Determine appropriate variable type based on query type
        if structured_response.query_type == "causal":
            # In causal analysis, employment income is typically an outcome/indicator
            employment_income_type = "causal-outcome"
        else:
            # In descriptive analysis, employment income is a main variable
            employment_income_type = "descriptive-main"
        
        # Add employment income as conceptual variable
        employment_income_var = structured_response.add_conceptual_variable("Employment income", employment_income_type)
        
        # Search for employment income variables in PIT_ITR
        try:
            earnings_results = search_engine.search_variables(
                "wages and salary", 
                "ato", 
                boost_datasets=relevant_datasets,
                use_openai_relevance=True,
                conceptual_variable="wages and salary"
            )
            
            if earnings_results:
                # Filter to only show PIT_ITR results
                pit_itr_results = [result for result in earnings_results 
                                 if result['row'].get('dataset_id', '').upper() == 'PIT_ITR']
                
                # Remove duplicates based on variable_name
                seen_variables = set()
                unique_results = []
                for result in pit_itr_results:
                    var_name = result['row'].get('variable_name', 'Unknown')
                    if var_name not in seen_variables:
                        seen_variables.add(var_name)
                        unique_results.append(result)
                
                # Add employment income variable matches to Response object
                for result in unique_results[:5]:  # Limit to top 5 results
                    row = result['row']
                    structured_response.add_variable_match(
                        "Employment income",
                        row.get('dataset_id', ''),
                        row.get('variable_name', ''),
                        row.get('description', ''),
                        result.get('openai_relevance', None)
                    )
                    
        except Exception as e:
            # If employment income search fails, still add the conceptual variable
            # but recommend PIT_ITR dataset
            structured_response.add_dataset_recommendations("Employment income", ["PIT_ITR"])

def capture_and_display_response(user_input, gpt_data, causal_data, relevant_datasets, dataset_results, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display, start_time, categorized=None):
    """Capture and display a complete response, storing it in session state."""
    response_data = {
        'user_input': user_input,
        'gpt_data': gpt_data,
        'causal_data': causal_data,
        'relevant_datasets': relevant_datasets,
        'dataset_results': dataset_results,
        'timestamp': time.time(),
        'start_time': start_time
    }
    
    # Create structured Response object for internal representation
    structured_response = None
    if categorized:
        structured_response = Response.from_analysis_data(
            user_input, gpt_data, causal_data, categorized, dataset_results
        )
        
        # Populate with actual variable search results
        populate_response_with_variables(
            structured_response, categorized, search_engine, search_filters, 
            query_analyzer, user_input, gpt_data.get('topic', ''), relevant_datasets, 
            gpt_data.get('is_longitudinal', False), dataframes
        )
        
        response_data['structured_response'] = structured_response
    
    # Store this response
    st.session_state.current_response = response_data
    
    # Store structured response separately for easy access
    if structured_response:
        st.session_state.structured_response = structured_response
    
    # Display the response
    with st.container():
        # Display the response content directly without header
        display_single_response(response_data, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display)
        
        # Add debug JSON output
        # if structured_response:
        #     st.markdown("---")
        #     st.markdown("## Debug: Structured Response JSON")
        #     
        #     # Show JSON without metadata first
        #     with st.expander("Response JSON (Clean)", expanded=False):
        #         json_output = structured_response.to_json(include_metadata=False)
        #         st.code(json_output, language='json')
        #     
        #     # Show JSON with metadata
        #     with st.expander("Response JSON (With Metadata)", expanded=False):
        #         json_output_meta = structured_response.to_json(include_metadata=True)
        #         st.code(json_output_meta, language='json')
        #     
        #     # Show summary stats
        #     with st.expander("Response Summary", expanded=False):
        #         st.write(f"**Query Type:** {structured_response.query_type}")
        #         st.write(f"**Number of Conceptual Variables:** {len(structured_response.conceptual_variables)}")
        #         st.write(f"**Relevant Datasets:** {len(structured_response.relevant_datasets)}")
        #         
        #         for i, var in enumerate(structured_response.conceptual_variables, 1):
        #             st.write(f"**{i}. {var.name}** ({var.variable_type}):")
        #             st.write(f"   - Actual variables: {len(var.actual_variables)}")
        #             st.write(f"   - Recommended datasets: {len(var.recommended_datasets)}")
        #             if var.actual_variables:
        #                 # Sort variables by relevance score in descending order for display
        #                 sorted_variables = sorted(
        #                     var.actual_variables,
        #                     key=lambda x: x.relevance_score or 0,
        #                     reverse=True
        #                 )
        #                 for j, match in enumerate(sorted_variables, 1):
        #                     relevance = f" (relevance: {match.relevance_score:.2f})" if match.relevance_score else ""
        #                     st.write(f"     {j}. {match.dataset_id}.{match.variable_name}{relevance}")
        #             if var.recommended_datasets:
        #                 st.write(f"     Recommended: {', '.join(var.recommended_datasets)}")

def substitute_acld_with_census(dataset_results, is_longitudinal):
    """Substitute ACLD with CENSUS when longitudinal analysis is not needed."""
    if is_longitudinal:
        return dataset_results  # Keep ACLD for longitudinal analysis
    
    substituted_results = []
    for result in dataset_results:
        row = result['row']
        dataset_id = row.get('dataset_id', '')
        
        if dataset_id.upper() == 'ACLD':
            # Get CENSUS info from datasets.csv
            try:
                import pandas as pd
                datasets_df = pd.read_csv('resources/datasets.csv')
                census_row_csv = datasets_df[datasets_df['dataset_id'] == 'CENSUS']
                
                if not census_row_csv.empty:
                    census_data = census_row_csv.iloc[0]
                    census_row = {
                        'dataset_id': 'CENSUS',
                        'dataset_name': census_data['dataset_name'],
                        'dataset_description': census_data['dataset_description']
                    }
                else:
                    # Fallback to hardcoded values
                    census_row = {
                        'dataset_id': 'CENSUS',
                        'dataset_name': 'Census of Population and Housing 2011',
                        'dataset_description': 'Complete population coverage from the 2011 Census, providing comprehensive demographic and socio-economic information.'
                    }
            except Exception:
                # Fallback to hardcoded values if CSV loading fails
                census_row = {
                    'dataset_id': 'CENSUS',
                    'dataset_name': 'Census of Population and Housing 2011',
                    'dataset_description': 'Complete population coverage from the 2011 Census, providing comprehensive demographic and socio-economic information.'
                }
            
            substituted_results.append({'row': census_row, 'score': result['score']})
        else:
            substituted_results.append(result)
    
    return substituted_results

def display_single_response(response_data, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display):
    """Display a single stored response."""
    user_input = response_data['user_input']
    gpt_data = response_data['gpt_data']
    causal_data = response_data['causal_data']
    relevant_datasets = response_data['relevant_datasets']
    dataset_results = response_data['dataset_results']
    start_time = response_data['start_time']
    timestamp = response_data['timestamp']
    
    # Display narrative introduction
    reformulated_question = gpt_data.get('reformulated_question', user_input)
    
    # Determine analysis type for narrative
    if causal_data['is_causal'] and causal_data['confidence'] > 0.6:
        analysis_type = "causal"
        analysis_description = "focusing on the relationships of cause and effect between variables"
    else:
        analysis_type = "descriptive" 
        analysis_description = "exploring descriptive patterns and characteristics"
    
    # Create narrative introduction
    narrative_intro = f"It seems like you're interested in {reformulated_question.lower()}. This question is **{analysis_type}**, {analysis_description}."
    
    # Add variables list if available
    variables = gpt_data.get('variables', [])
    if variables:
        # Remove parenthetical information from variables
        clean_variables = []
        for var in variables:
            # Remove text in parentheses (like "(demography)", "(employment)", etc.)
            clean_var = var.split('(')[0].strip()
            clean_variables.append(clean_var)
        
        # Create numbered list
        variables_list = "\n".join([f"{i+1}. {var}" for i, var in enumerate(clean_variables)])
        narrative_intro += f"\n\nYou may need to consider the following variables:\n{variables_list}"
    
    # Polish the narrative with OpenAI
    try:
        polished_narrative = query_analyzer.polish_narrative_intro(narrative_intro)
        st.markdown(polished_narrative)
    except Exception as e:
        # Fallback to original narrative if polishing fails
        st.markdown(narrative_intro)
    
    # Check relevance
    if gpt_data['relevance_score'] < 5:
        result_display.display_low_relevance()
        return
    
    # Display medical condition suggestion BEFORE variables if detected
    medical_info = gpt_data.get('medical_condition_detected', {})
    if medical_info.get('is_medical', False):
        result_display.display_medical_condition_suggestion(
            medical_info,
            dataframes['datasets'],
            relevant_datasets,
            query_analyzer
        )
    
    # Show loading indicator while collecting datasets and variables
    loading_container = ui.display_loading_indicator()
    
    # Process variables based on whether query is causal and collect all dataset IDs
    variables = gpt_data['variables']
    all_dataset_ids = []
    
    if causal_data['is_causal'] and causal_data['confidence'] > 0.6:
        
        # Categorize variables for causal analysis
        categorized = causal_analyzer.categorize_variables(
            variables, 
            user_input, 
            gpt_data['topic']
        )
        
        # Collect datasets from variables WITHOUT displaying them yet
        variable_dataset_ids = result_display.collect_variable_datasets(
            categorized,
            search_engine,
            search_filters,
            gpt_data['topic'],
            relevant_datasets,
            user_input,
            query_analyzer,
            analysis_type="causal"
        )
        if variable_dataset_ids:
            all_dataset_ids.extend(variable_dataset_ids)
        
    else:
        
        # Categorize variables for descriptive analysis
        categorized = causal_analyzer.descriptive_categorize_variables(
            variables,
            user_input,
            gpt_data['topic']
        )
        
        # Collect datasets from variables WITHOUT displaying them yet
        variable_dataset_ids = result_display.collect_variable_datasets(
            categorized,
            search_engine,
            search_filters,
            gpt_data['topic'],
            relevant_datasets,
            user_input,
            query_analyzer,
            analysis_type="descriptive"
        )
        if variable_dataset_ids:
            all_dataset_ids.extend(variable_dataset_ids)
    
    # Now collect all datasets from both initial search and variables
    dataset_ids_from_results = set()
    if dataset_results:
        for result in dataset_results:
            row = result['row']
            # Skip ACLD for healthcare (this is now less likely since we substitute ACLD)
            if not (gpt_data['topic'] == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                if row.get('dataset_id'):
                    dataset_ids_from_results.add(row['dataset_id'])
    
    # Combine dataset IDs from results and variables
    all_combined_dataset_ids = list(dataset_ids_from_results)
    if all_dataset_ids:
        # Flatten the list of sets from variables and add to combined list
        for dataset_id_set in all_dataset_ids:
            if isinstance(dataset_id_set, set):
                all_combined_dataset_ids.extend(list(dataset_id_set))
            else:
                all_combined_dataset_ids.append(dataset_id_set)
    
    # Remove duplicates for unified dataset list
    unique_dataset_ids = list(set(all_combined_dataset_ids))
    
    # Display combined datasets and variables
    if dataset_results or unique_dataset_ids:
        result_display.display_datasets_and_variables(
            dataset_results,
            categorized if 'categorized' in locals() else None,
            search_engine,
            search_filters,
            gpt_data['topic'],
            ui,
            relevant_datasets,
            user_input,
            query_analyzer,
            causal_data,
            unique_dataset_ids,
            dataframes,
            loading_container,  # Pass loading container to clear it when display starts
            gpt_data.get('is_longitudinal', False)  # Pass longitudinal flag
        )
    
    # Display execution time
    execution_time = timestamp - start_time
    st.caption(f"Execution time: {execution_time:.2f} seconds")

def main():
    """Main application function."""
    # Initialize UI components
    ui = UIComponents()
    result_display = ResultDisplay()
    
    # Display header and image
    ui.display_header()
    #ui.display_image("principles.png")
    
    # Initialize system
    try:
        data_loader, search_engine, dataframes, query_analyzer = initialize_system()
        causal_analyzer = CausalAnalyzer()
        search_filters = SearchFilters()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()
    
    # Initialize user_input
    user_input = None
    
    # Query input
    user_input = st.text_input("Ask a question:")
    
    # Display previous response if exists
    if st.session_state.current_response and not user_input:
        with st.container():
            display_single_response(
                st.session_state.current_response, 
                dataframes, search_engine, search_filters, 
                query_analyzer, causal_analyzer, ui, result_display
            )
            
            # Add new query option
            st.markdown("---")
            st.write("You can submit a new query by opening a new chat above.")
    
    if user_input:
        start_time = time.time()
        
        try:
            # Analyze query with LLM
            gpt_data = query_analyzer.analyze_query(user_input)
            
            # Check relevance
            if gpt_data['relevance_score'] < 5:
                # For low relevance, just display the message without storing
                result_display.display_low_relevance()
            else:
                # Check if query is causal
                causal_data = causal_analyzer.is_causal_query(user_input)
                
                # Search for relevant datasets
                dataset_results = search_engine.search_datasets(user_input)
                
                # Apply ACLD to CENSUS substitution if not longitudinal
                is_longitudinal = gpt_data.get('is_longitudinal', False)
                dataset_results = substitute_acld_with_census(dataset_results, is_longitudinal)
                
                relevant_datasets = []
                for result in dataset_results:
                    row = result['row']
                    # Skip ACLD for healthcare (this is now less likely to trigger since we substitute ACLD)
                    if not (gpt_data['topic'] == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                        relevant_datasets.append(row['dataset_name'])
                
                # Handle medical condition detection and add PBS and MBS to dataset_results
                medical_info = gpt_data.get('medical_condition_detected', {})
                if medical_info.get('is_medical', False):
                    # Add PBS dataset to dataset_results if not already present
                    pbs_row = dataframes['datasets'][dataframes['datasets']["dataset_id"].str.upper() == "PBS"]
                    if not pbs_row.empty:
                        pbs_data = pbs_row.iloc[0]
                        pbs_name = pbs_data['dataset_name']
                        # Check if PBS is not already in dataset_results
                        pbs_exists = any(result['row']['dataset_name'] == pbs_name for result in dataset_results)
                        if not pbs_exists and pbs_name not in relevant_datasets:
                            dataset_results.append({'row': pbs_data, 'score': 1.0})
                            relevant_datasets.append(pbs_name)
                    
                    # Add MBS dataset to dataset_results if not already present
                    mbs_row = dataframes['datasets'][dataframes['datasets']["dataset_id"].str.upper() == "MBS"]
                    if not mbs_row.empty:
                        mbs_data = mbs_row.iloc[0]
                        mbs_name = mbs_data['dataset_name']
                        # Check if MBS is not already in dataset_results
                        mbs_exists = any(result['row']['dataset_name'] == mbs_name for result in dataset_results)
                        if not mbs_exists and mbs_name not in relevant_datasets:
                            dataset_results.append({'row': mbs_data, 'score': 1.0})
                            relevant_datasets.append(mbs_name)
                
                # Handle employment-related queries and add PIT_ITR to dataset_results
                # Check if query contains employment-related keywords
                employment_keywords = ['employment', 'job', 'work', 'income', 'salary', 'wage', 'earn']
                query_lower = user_input.lower()
                if any(keyword in query_lower for keyword in employment_keywords):
                    # Add PIT_ITR dataset to dataset_results if not already present
                    pit_itr_row = dataframes['datasets'][dataframes['datasets']["dataset_id"].str.upper() == "PIT_ITR"]
                    if not pit_itr_row.empty:
                        pit_itr_data = pit_itr_row.iloc[0]
                        pit_itr_name = pit_itr_data['dataset_name']
                        # Check if PIT_ITR is not already in dataset_results
                        pit_itr_exists = any(result['row']['dataset_name'] == pit_itr_name for result in dataset_results)
                        if not pit_itr_exists and pit_itr_name not in relevant_datasets:
                            dataset_results.append({'row': pit_itr_data, 'score': 1.0})
                            relevant_datasets.append(pit_itr_name)
                
                # Categorize variables for Response object
                variables = gpt_data.get('variables', [])
                categorized = None
                if variables:
                    if causal_data['is_causal'] and causal_data['confidence'] > 0.6:
                        # Categorize variables for causal analysis
                        categorized = causal_analyzer.categorize_variables(
                            variables, 
                            user_input, 
                            gpt_data['topic']
                        )
                    else:
                        # Categorize variables for descriptive analysis
                        categorized = causal_analyzer.descriptive_categorize_variables(
                            variables,
                            user_input,
                            gpt_data['topic']
                        )
                
                # Store and display the complete response
                capture_and_display_response(
                    user_input, gpt_data, causal_data, relevant_datasets, 
                    dataset_results, dataframes, search_engine, search_filters, 
                    query_analyzer, causal_analyzer, ui, result_display, start_time, categorized
                )
                
                # Add new query option after response
                st.markdown("---")
                st.write("You can submit a new query by opening a new chat above.")
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
    

if __name__ == "__main__":
    main()