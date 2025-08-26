import streamlit as st
import time
from config import TEXT_STREAM_DELAY

class UIComponents:
    @staticmethod
    def display_header():
        """Display the application header."""
        st.markdown("""
            <h1 style='text-align: center; margin-bottom: 0;'>PLIDA Helper</h1>
            <h3 style='text-align: center; margin-top: 0;'>Navigating Australia's Premier Data Asset</h3>
        """, unsafe_allow_html=True)
        
        # Add usage instructions
        st.markdown("---")
        st.markdown("### 📋 How to Use PLIDA Helper")
        
        st.markdown("""
        **1. Write your research question** - Be specific and brief. Include the key concepts, population of interest, and what you want to study.
        
        **2. Review the results** - The system will display PLIDA datasets and variables relevant to your question, organized by analysis type and variable categories.
        
        **3. Important notes:**
        - The Helper errs on the side of inclusion: not all variables displayed may be directly relevant to your query, but none are hallucinated. All results come from actual PLIDA datasets.
        - The Helper only considers administrative data and survey data. ABS surveys are not included.
        """)
        
        # Add examples
        with st.expander("💡 Example Research Questions", expanded=False):
            st.markdown("""
            **Good examples:**
            - "Impact of NDIS on mental healthcare utilization"
            - "Relationship between early childhood education and later academic outcomes"
            - "Employment outcomes for refugees in Australia"
            - "Factors affecting aged care service usage"
            
            **Less effective:**
            - "Tell me about health" (too broad)
            - "NDIS" (too vague - specify what aspect you're studying)
            - "Show me all education data" (not a research question)
            """)
        
        st.markdown("---")
    
    @staticmethod
    def display_image(image_path):
        """Display an image with error handling."""
        try:
            st.image(image_path, use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image '{image_path}' not found. Please ensure the file exists.")
    
    @staticmethod
    def stream_text(container, text, delay=TEXT_STREAM_DELAY):
        """Stream text character by character for better UX."""
        current_text = ""
        for char in text:
            current_text += char
            container.markdown(current_text)
            time.sleep(delay)
    
    @staticmethod
    def display_execution_time(start_time, end_time):
        """Display execution time."""
        st.write(f"Execution time: {end_time - start_time:.2f} seconds")