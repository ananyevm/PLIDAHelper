import time
import streamlit as st
from config import DATASETS_PATH, VARIABLES_PATH
from data import DataLoader
from search import IndexBuilder, SemanticSearchEngine, SearchFilters
from llm import QueryAnalyzer, CausalAnalyzer
from ui import UIComponents, ResultDisplay
# EmbeddingManager is initialized within other components

# Initialize session state for responses
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

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

def capture_and_display_response(user_input, gpt_data, causal_data, relevant_datasets, dataset_results, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display, start_time):
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
    
    # Store this response
    st.session_state.current_response = response_data
    
    # Display the response
    with st.container():
        # Display the response content directly without header
        display_single_response(response_data, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display)

def substitute_acld_with_census(dataset_results, is_longitudinal):
    """Substitute ACLD with CENSUS when longitudinal analysis is not needed."""
    if is_longitudinal:
        return dataset_results  # Keep ACLD for longitudinal analysis
    
    substituted_results = []
    for result in dataset_results:
        row = result['row']
        dataset_id = row.get('dataset_id', '')
        
        if dataset_id.upper() == 'ACLD':
            # Try to find CENSUS equivalent
            # For now, we'll replace ACLD entry with a CENSUS entry
            # In a real implementation, you might want to load CENSUS data from dataframes
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
    if gpt_data['relevance_score'] < 6:
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
            if gpt_data['relevance_score'] < 6:
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
                
                # Store and display the complete response
                capture_and_display_response(
                    user_input, gpt_data, causal_data, relevant_datasets, 
                    dataset_results, dataframes, search_engine, search_filters, 
                    query_analyzer, causal_analyzer, ui, result_display, start_time
                )
                
                # Add new query option after response
                st.markdown("---")
                st.write("You can submit a new query by opening a new chat above.")
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
    

if __name__ == "__main__":
    main()