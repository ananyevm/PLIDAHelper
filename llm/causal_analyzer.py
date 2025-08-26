import json
import streamlit as st
from .client import LLMClient

class CausalAnalyzer:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def is_causal_query(self, user_input):
        """Determine if the query involves causal relationships."""
        prompt = self._build_causal_detection_prompt(user_input)
        system_prompt = "You are an expert in causal inference and research methodology. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_causal_response(response)
    
    def categorize_variables(self, variables, user_input, topic):
        """Categorize variables into causal, outcome, and control variables."""
        prompt = self._build_categorization_prompt(variables, user_input, topic)
        system_prompt = "You are an expert in causal inference and research design. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_categorization_response(response)
    
    def _build_causal_detection_prompt(self, user_input):
        """Build prompt to detect if query is causal."""
        return (
            f"Analyze the following research question: '{user_input}'\n\n"
            "Determine if this question involves causal relationships or causal inference.\n"
            "A causal question typically asks about:\n"
            "- The effect or impact of one variable on another\n"
            "- Whether X causes Y\n"
            "- The relationship between variables\n"
            "- How changes in one factor influence another\n"
            "- Comparisons between groups or interventions\n\n"
            "Return a JSON object with:\n"
            "- 'is_causal': boolean (true if the question involves causal relationships)\n"
            "- 'confidence': float (0.0 to 1.0, your confidence in this assessment)\n"
            "- 'reasoning': string (brief explanation of why it is or isn't causal)\n\n"
            "Example: {'is_causal': true, 'confidence': 0.85, 'reasoning': 'Question asks about the impact of education on income'}"
        )
    
    def _build_categorization_prompt(self, variables, user_input, topic):
        """Build prompt to categorize variables."""
        variables_str = "\n".join([f"- {var}" for var in variables])
        
        return (
            f"For the research question: '{user_input}'\n"
            f"Topic: {topic}\n\n"
            f"Categorize the following variables into causal analysis categories:\n{variables_str}\n\n"
            "Categories:\n"
            "1. **Potential Causal Variables** (Treatment/Exposure): Variables that might cause or influence the outcome\n"
            "2. **Outcome Variables** (Dependent): Variables being affected or measured as results\n"
            "3. **Control Variables** (Confounders/Covariates): Variables that need to be controlled for valid causal inference\n\n"
            "Return a JSON object with three lists:\n"
            "- 'causal_variables': list of potential causal variables\n"
            "- 'outcome_variables': list of outcome variables\n"
            "- 'control_variables': list of control variables\n\n"
            "Each variable should appear in exactly one category.\n"
            "Example: {'causal_variables': ['Education level'], 'outcome_variables': ['Income'], 'control_variables': ['Age', 'Gender']}"
        )
    
    def _parse_causal_response(self, response):
        """Parse causal detection response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response)
            
            return {
                'is_causal': data.get('is_causal', False),
                'confidence': data.get('confidence', 0.0),
                'reasoning': data.get('reasoning', ''),
                'raw_response': response
            }
            
        except Exception as e:
            st.error(f"Error parsing causal detection response: {e}")
            return {
                'is_causal': False,
                'confidence': 0.0,
                'reasoning': 'Error in analysis',
                'raw_response': response
            }
    
    def _parse_categorization_response(self, response):
        """Parse variable categorization response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response)
            
            return {
                'causal_variables': data.get('causal_variables', []),
                'outcome_variables': data.get('outcome_variables', []),
                'control_variables': data.get('control_variables', []),
                'raw_response': response
            }
            
        except Exception as e:
            st.error(f"Error parsing categorization response: {e}")
            return {
                'causal_variables': [],
                'outcome_variables': [],
                'control_variables': [],
                'raw_response': response
            }
    
    def descriptive_categorize_variables(self, variables, user_input, topic):
        """Categorize variables for descriptive analysis into main variables and subgroups."""
        prompt = self._build_descriptive_categorization_prompt(variables, user_input, topic)
        system_prompt = "You are an expert in research design and data analysis. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_descriptive_categorization_response(response)
    
    def _build_descriptive_categorization_prompt(self, variables, user_input, topic):
        """Build prompt to categorize variables for descriptive analysis."""
        variables_str = "\n".join([f"- {var}" for var in variables])
        
        return (
            f"For the descriptive research question: '{user_input}'\n"
            f"Topic: {topic}\n\n"
            f"Categorize the following variables for descriptive analysis:\n{variables_str}\n\n"
            "Categories:\n"
            "1. **Main Variables**: Primary variables that directly address the research question (e.g., the main outcomes or characteristics being described)\n"
            "2. **Subgroups**: Variables used to break down or segment the analysis (e.g., demographic groups, geographic areas, time periods)\n\n"
            "Return a JSON object with two lists:\n"
            "- 'main_variables': list of main variables that are central to the research question\n"
            "- 'subgroups': list of variables used for segmentation or subgroup analysis\n\n"
            "Each variable should appear in exactly one category.\n"
            "Example: {'main_variables': ['Unemployment rate', 'Employment status'], 'subgroups': ['Age group', 'Gender', 'State', 'Year']}"
        )
    
    def _parse_descriptive_categorization_response(self, response):
        """Parse descriptive variable categorization response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response)
            
            return {
                'main_variables': data.get('main_variables', []),
                'subgroups': data.get('subgroups', []),
                'raw_response': response
            }
            
        except Exception as e:
            st.error(f"Error parsing descriptive categorization response: {e}")
            return {
                'main_variables': [],
                'subgroups': [],
                'raw_response': response
            }