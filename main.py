import time
import streamlit as st
from config import DATASETS_PATH, VARIABLES_PATH
from data import DataLoader
from search import IndexBuilder, SemanticSearchEngine, SearchFilters
from llm import QueryAnalyzer, CausalAnalyzer
from ui import UIComponents, ResultDisplay
from embeddings import EmbeddingManager

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
    
    # Create search engine
    search_engine = SemanticSearchEngine(indices, dataframes)
    
    return data_loader, search_engine, dataframes

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
        st.markdown("## 🔍 **Query Results**")
        
        # Display the response content
        display_single_response(response_data, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display)

def display_single_response(response_data, dataframes, search_engine, search_filters, query_analyzer, causal_analyzer, ui, result_display):
    """Display a single stored response."""
    user_input = response_data['user_input']
    gpt_data = response_data['gpt_data']
    causal_data = response_data['causal_data']
    relevant_datasets = response_data['relevant_datasets']
    dataset_results = response_data['dataset_results']
    start_time = response_data['start_time']
    timestamp = response_data['timestamp']
    
    # Display query
    st.markdown(f"**Research Question:** {user_input}")
    
    # Check relevance
    if gpt_data['relevance_score'] < 6:
        result_display.display_low_relevance()
        return
    
    # Display dataset results
    if dataset_results:
        st.subheader("Most Relevant Datasets")
        for result in dataset_results:
            row = result['row']
            score = result['score']
            
            # Skip ACLD for healthcare
            if not (gpt_data['topic'] == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                from utils import truncate_description
                truncated_desc = truncate_description(row['dataset_description'])
                dataset_text = (
                    f"- **Dataset:** {row['dataset_id']} | "
                    f"**Name:** {row['dataset_name']} | "
                    f"**Description:** {truncated_desc} | "
                    f"**Score:** {score:.3f}"
                )
                st.markdown(dataset_text)
    
    # Display medical condition suggestion BEFORE variables if detected
    if gpt_data.get('medical_condition_detected'):
        result_display.display_medical_condition_suggestion(
            gpt_data['medical_condition_detected'],
            dataframes['datasets'],
            relevant_datasets
        )
    
    # Process variables based on whether query is causal
    variables = gpt_data['variables']
    
    if causal_data['is_causal'] and causal_data['confidence'] > 0.6:
        # Display causal analysis info
        st.info(f"📊 Causal Analysis Detected (Confidence: {causal_data['confidence']:.0%})")
        st.caption(f"*{causal_data['reasoning']}*")
        
        # Categorize variables for causal analysis
        categorized = causal_analyzer.categorize_variables(
            variables, 
            user_input, 
            gpt_data['topic']
        )
        
        # Display categorized variables with dataset prioritization
        result_display.display_causal_variables(
            categorized,
            search_engine,
            search_filters,
            gpt_data['topic'],
            ui,
            relevant_datasets,
            user_input,
            query_analyzer
        )
        
    else:
        # Descriptive analysis (non-causal)
        st.info("📊 Descriptive Analysis Detected")
        st.caption("*This question seeks to describe patterns and characteristics rather than causal relationships*")
        
        # Categorize variables for descriptive analysis
        categorized = causal_analyzer.descriptive_categorize_variables(
            variables,
            user_input,
            gpt_data['topic']
        )
        
        # Display categorized variables for descriptive analysis with dataset prioritization
        result_display.display_descriptive_variables(
            categorized,
            search_engine,
            search_filters,
            gpt_data['topic'],
            ui,
            relevant_datasets,
            user_input,
            query_analyzer
        )
    
    # Display geographic analysis suggestion AFTER everything else
    if gpt_data.get('geographic_analysis_detected'):
        result_display.display_geographic_analysis_suggestion(
            gpt_data['geographic_analysis_detected'],
            search_engine,
            relevant_datasets,
            user_input,
            query_analyzer
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
        data_loader, search_engine, dataframes = initialize_system()
        query_analyzer = QueryAnalyzer()
        causal_analyzer = CausalAnalyzer()
        embedding_manager = EmbeddingManager()
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
            st.markdown("## 🔍 **Previous Query Results**")
            display_single_response(
                st.session_state.current_response, 
                dataframes, search_engine, search_filters, 
                query_analyzer, causal_analyzer, ui, result_display
            )
            
            # Add new query option
            st.markdown("---")
            st.info("You can submit a new query by opening a new chat above.")
    
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
                relevant_datasets = []
                for result in dataset_results:
                    row = result['row']
                    # Skip ACLD for healthcare
                    if not (gpt_data['topic'] == "healthcare" and row["dataset_id"].upper() == "ACLD"):
                        relevant_datasets.append(row['dataset_name'])
                
                # Handle medical condition detection and add PBS if needed
                if gpt_data.get('medical_condition_detected'):
                    pbs_row = dataframes['datasets'][dataframes['datasets']["dataset_id"].str.upper() == "PBS"]
                    if not pbs_row.empty:
                        pbs_name = pbs_row.iloc[0]['dataset_name']
                        if pbs_name not in relevant_datasets:
                            relevant_datasets.append(pbs_name)
                
                # Store and display the complete response
                capture_and_display_response(
                    user_input, gpt_data, causal_data, relevant_datasets, 
                    dataset_results, dataframes, search_engine, search_filters, 
                    query_analyzer, causal_analyzer, ui, result_display, start_time
                )
                
                # Add new query option after response
                st.markdown("---")
                st.info("You can submit a new query by opening a new chat above.")
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
    

if __name__ == "__main__":
    main()