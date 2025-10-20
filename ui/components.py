import streamlit as st
import time
from config import TEXT_STREAM_DELAY

class UIComponents:
    @staticmethod
    def display_header():
        """Display the application header."""
        st.markdown("""
            <h1 style='text-align: center; margin-bottom: 0;'>PLIDA Helper</h1>
            <h3 style='text-align: center; margin-top: 0;'>AI Assistant for Navigating Australia's Premier Data Asset</h3>
        """, unsafe_allow_html=True)
        
        # Add usage instructions
        st.markdown("---")
        st.markdown("### 📋 How to Use PLIDA Helper")
        
        st.markdown("""
        **1. Enter the question you would like to explore with PLIDA** - Be specific and brief. Include the key concepts, population of interest.
        
        **2. Review the results** - The system will display PLIDA datasets and variables relevant to your question, organized by analysis type and variable categories.
        
        **3. Notes:**
        - The Helper errs on the side of inclusion: not all variables displayed may be directly relevant to your query, but none are hallucinated. All results come from actual PLIDA datasets.
        - The Helper only considers administrative data and Census data. Most of ABS surveys are not included.
        - The Helper currently draws only on information that is publicly available, which limits the extent of its expertise.
        - At this point, the Helper is instructed not to remember past interactions. This helps the Helper to stay accurate and focused on the current context.           
        """)
        
        # Add examples
        with st.expander("💡 Example Questions", expanded=False):
            st.markdown("""
            **Good examples:**
            - "What is the average life expectancy of Aboriginal men living in Darwin?"
            - "Which industries attract most STEM degree graduates?" 
            - "Impact of NDIS participation on the utilization of mental healthcare."

            
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
    
    @staticmethod
    def display_loading_indicator():
        """Display a fading loading indicator for dataset and variable collection."""
        # First add the CSS styles
        st.markdown("""
            <style>
            @keyframes fade-in-out {
                0% { opacity: 0.3; }
                50% { opacity: 1.0; }
                100% { opacity: 0.3; }
            }
            
            .loading-indicator {
                text-align: center;
                font-size: 18px;
                color: #1f77b4;
                font-weight: 500;
                animation: fade-in-out 2s infinite;
                padding: 20px;
                margin: 20px 0;
                border: 2px dashed #1f77b4;
                border-radius: 10px;
                background-color: rgba(31, 119, 180, 0.05);
                min-height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                width: 100%;
                box-sizing: border-box;
            }
            
            </style>
        """, unsafe_allow_html=True)
        
        # Create a stable container that won't cause layout shifts
        loading_container = st.empty()
        
        # Create a wrapper div to ensure stable positioning
        loading_container.markdown("""
            <div class="loading-wrapper" style="min-height: 100px; position: relative;">
                <div class="loading-indicator">
                    🔍 Collecting PLIDA datasets and variables
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        return loading_container  # Return container that can be cleared later