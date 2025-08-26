import time
import streamlit as st
from config import DATASETS_PATH, VARIABLES_PATH
from data import DataLoader
from search import IndexBuilder, SemanticSearchEngine, SearchFilters
from llm import QueryAnalyzer, CausalAnalyzer
from ui import UIComponents, ResultDisplay
from embeddings import EmbeddingManager

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
    
    # Get user input
    user_input = st.text_input("Ask a question:")
    
    if user_input:
        start_time = time.time()
        
        try:
            # Analyze query with LLM
            gpt_data = query_analyzer.analyze_query(user_input)
            
            # Check relevance
            if gpt_data['relevance_score'] < 6:
                result_display.display_low_relevance()
            else:
                # Check if query is causal
                causal_data = causal_analyzer.is_causal_query(user_input)
                
                # Search for relevant datasets
                dataset_results = search_engine.search_datasets(user_input)
                relevant_datasets = result_display.display_dataset_results(
                    user_input, 
                    dataset_results, 
                    gpt_data['topic'],
                    dataframes['datasets'],
                    search_engine
                )
                
                # Display medical condition suggestion BEFORE variables if detected
                if gpt_data.get('medical_condition_detected'):
                    relevant_datasets = result_display.display_medical_condition_suggestion(
                        gpt_data['medical_condition_detected'],
                        dataframes['datasets'],
                        relevant_datasets
                    )
                
                # Display geographic analysis suggestion AFTER medical but BEFORE variables
                if gpt_data.get('geographic_analysis_detected'):
                    result_display.display_geographic_analysis_suggestion(
                        gpt_data['geographic_analysis_detected'],
                        search_engine,
                        relevant_datasets,
                        user_input,
                        query_analyzer
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
                    
                    # Display debug info for causal analysis
                    with st.expander("Causal Analysis Details (Debug)"):
                        st.json(causal_data)
                        st.json(categorized)
                    
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
                    
                    # Display debug info for descriptive analysis
                    with st.expander("Descriptive Analysis Details (Debug)"):
                        st.json(categorized)
                
                # Display debug info
                result_display.display_debug_info(gpt_data)
            
            # Display execution time
            ui.display_execution_time(start_time, time.time())
            
        except Exception as e:
            st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()