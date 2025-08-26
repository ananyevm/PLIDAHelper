import json
import pandas as pd
import streamlit as st
from .client import LLMClient
from config import VARIABLES_PATH, DATASETS_PATH

class QueryAnalyzer:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def analyze_query(self, user_input):
        """Analyze user query for relevance and extract variables."""
        prompt = self._build_analysis_prompt(user_input)
        system_prompt = "You are a helpful assistant. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        result = self._parse_response(response)
        
        # Check if query involves medical conditions
        medical_info = self.detect_medical_condition_query(user_input)
        result['medical_condition_detected'] = medical_info
        
        # Check if query involves geographic analysis
        geographic_info = self.detect_geographic_analysis_query(user_input)
        result['geographic_analysis_detected'] = geographic_info
        
        return result
    
    def detect_medical_condition_query(self, user_input):
        """Detect if query involves medical conditions and suggest PBS dataset."""
        prompt = self._build_medical_detection_prompt(user_input)
        system_prompt = "You are a healthcare data expert. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_medical_response(response)
    
    def detect_geographic_analysis_query(self, user_input):
        """Detect if query involves geographic analysis."""
        prompt = self._build_geographic_detection_prompt(user_input)
        system_prompt = "You are a geographic and spatial analysis expert. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_geographic_response(response)
    
    def suggest_geographic_variables(self, user_input, geographic_info, relevant_datasets):
        """Use OpenAI to suggest relevant geographic variables for the analysis."""
        if not geographic_info.get('is_geographic', False):
            return []
        
        prompt = self._build_geographic_variables_prompt(user_input, geographic_info, relevant_datasets)
        system_prompt = "You are a geographic data analysis expert. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_geographic_variables_response(response)
    
    def check_population_match(self, user_input, variable_results, search_engine=None):
        """Check if variable datasets match the population in the user query."""
        filtered_results = []
        
        # Load datasets information
        try:
            datasets_df = pd.read_csv(DATASETS_PATH)
        except Exception as e:
            st.warning(f"Could not load datasets for population matching: {e}")
            return variable_results
        
        for result in variable_results:
            try:
                # Get dataset information
                dataset_name = result['row'].get('dataset', '')
                dataset_id = result['row'].get('dataset_id', '')
                
                # Find dataset description
                dataset_info = datasets_df[
                    (datasets_df['dataset_name'].str.contains(dataset_name, case=False, na=False)) |
                    (datasets_df['dataset_id'] == dataset_id)
                ]
                
                if not dataset_info.empty:
                    dataset_description = dataset_info.iloc[0]['dataset_description']
                    
                    # Check population match
                    population_match = self._assess_population_match(
                        user_input, dataset_name, dataset_description
                    )
                    
                    # Add population match information to result
                    result['population_match'] = population_match
                    
                    # Filter based on population match
                    if population_match['match'] != 'no':
                        filtered_results.append(result)
                else:
                    # If dataset not found, keep with unknown status
                    result['population_match'] = {
                        'match': 'unknown',
                        'reasoning': 'Dataset description not found'
                    }
                    filtered_results.append(result)
                    
            except Exception as e:
                # If error occurs, keep the result
                result['population_match'] = {
                    'match': 'error',
                    'reasoning': f'Error assessing population match: {str(e)}'
                }
                filtered_results.append(result)
        
        # Check if we should provide CORE alternatives
        should_show_core_alternatives = False
        if search_engine is not None:
            if len(filtered_results) == 0:
                # No matches at all - show CORE alternatives
                st.info("🔍 No population matches found. Searching CORE dataset for relevant variables...")
                should_show_core_alternatives = True
            else:
                # Check if any results have population warnings
                has_warnings = any(
                    result.get('population_match', {}).get('match') in ['maybe', 'error'] 
                    for result in filtered_results
                )
                if has_warnings:
                    st.info("⚠️ Some variables have population concerns. Also showing CORE demographic alternatives...")
                    should_show_core_alternatives = True
        
        # Add CORE alternatives if needed
        if should_show_core_alternatives:
            core_results = self._search_core_fallback(user_input, search_engine)
            if core_results:
                # Add a separator/header for CORE results
                for result in core_results:
                    result['is_core_alternative'] = True
                filtered_results.extend(core_results)
        
        return filtered_results
    
    def _assess_population_match(self, user_input, dataset_name, dataset_description):
        """Use OpenAI to assess if dataset population matches user query population."""
        prompt = self._build_population_match_prompt(user_input, dataset_name, dataset_description)
        system_prompt = "You are an expert in Australian administrative datasets and population analysis. Return only valid JSON."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            return self._parse_population_match_response(response)
        except Exception as e:
            return {
                'match': 'error',
                'reasoning': f'Error assessing population match: {str(e)}',
                'confidence': 0.0
            }
    
    def _build_population_match_prompt(self, user_input, dataset_name, dataset_description):
        """Build prompt to assess if dataset population matches user query population."""
        return (
            f"Research Question: '{user_input}'\n\n"
            f"Dataset Name: {dataset_name}\n"
            f"Dataset Description: {dataset_description}\n\n"
            "Analyze whether the population described in this dataset matches the target population implied in the research question.\n\n"
            "Consider:\n"
            "- **Age groups**: Does the dataset cover the age range needed for the research?\n"
            "- **Geographic scope**: Does the dataset cover the required geographic area?\n"
            "- **Population subset**: Does the dataset include or exclude relevant population groups?\n"
            "- **Eligibility criteria**: Are there restrictions that might exclude the target population?\n"
            "- **Temporal coverage**: Does the time period align with the research needs?\n\n"
            "Examples of mismatches:\n"
            "- Asking about 'all Australians' but dataset only covers 'university students'\n"
            "- Asking about 'children' but dataset only covers 'adults aged 18+'\n"
            "- Asking about 'employment' but dataset only covers 'retirees'\n"
            "- Asking about 'national trends' but dataset only covers 'one state'\n"
            "- Asking about 'disability services (NDIS)' but dataset only covers 'vocational education students'\n"
            "- Asking about 'healthcare utilisation' but dataset only covers 'school enrollment'\n"
            "- Asking about 'aged care' but dataset only covers 'young job seekers'\n\n"
            "Be strict about population alignment. If the research question targets a specific service population (e.g., disability, healthcare, aged care) and the dataset covers a completely different population (e.g., education, employment training), this should be marked as 'no' match.\n\n"
            "Return a JSON object with:\n"
            "- 'match': string ('yes' if populations align well, 'no' if clear mismatch, 'maybe' if partial overlap or uncertain)\n"
            "- 'reasoning': string explaining the assessment\n"
            "- 'confidence': float (0.0 to 1.0, confidence in this assessment)\n\n"
            "Example: {'match': 'maybe', 'reasoning': 'Dataset covers university students but research asks about all young adults - there is overlap but not complete coverage', 'confidence': 0.8}"
        )
    
    def _parse_population_match_response(self, response):
        """Parse and validate the population match response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_population_match_response()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate match value
            match_value = data.get('match', 'maybe').lower()
            if match_value not in ['yes', 'no', 'maybe']:
                match_value = 'maybe'
            
            return {
                'match': match_value,
                'reasoning': data.get('reasoning', 'No reasoning provided'),
                'confidence': float(data.get('confidence', 0.0))
            }
            
        except Exception as e:
            return self._default_population_match_response()
    
    def _default_population_match_response(self):
        """Return default population match response when parsing fails."""
        return {
            'match': 'maybe',
            'reasoning': 'Unable to assess population match',
            'confidence': 0.0
        }
    
    def _search_core_fallback(self, user_input, search_engine):
        """Search CORE dataset variables when no population matches are found."""
        try:
            # TEMPORARY: Force manual search for debugging
            st.info("🔧 Using manual CORE search for better age matching")
            return self._fallback_manual_core_search(user_input)
            
            # Try to use the existing semantic search engine to search CORE variables
            # First, let's see if we can use the plida4 index if it exists
            if 'plida4' in search_engine.indices:
                # Use semantic search on plida4 index with CORE filtering
                all_results = search_engine.search_variables(user_input, 'plida4', top_k=50)
                
                # Filter for CORE dataset only
                core_results = []
                for result in all_results:
                    if result['row'].get('dataset_id') == 'CORE':
                        # Add population match info
                        result['population_match'] = {
                            'match': 'yes',
                            'reasoning': 'CORE dataset contains demographic variables that typically match most populations',
                            'confidence': 0.8
                        }
                        core_results.append(result)
                
                # Return top 3 CORE results
                top_results = core_results[:3]
                
                if top_results:
                    st.success(f"Found {len(top_results)} relevant variables from CORE dataset")
                else:
                    st.warning("No matching variables found in CORE dataset using semantic search")
                
                return top_results
            else:
                # Fallback to manual text matching if plida4 index doesn't exist
                return self._fallback_manual_core_search(user_input)
                
        except Exception as e:
            st.error(f"Error with semantic search, trying manual fallback: {e}")
            return self._fallback_manual_core_search(user_input)
    
    def _fallback_manual_core_search(self, user_input):
        """Manual fallback search for CORE variables with improved matching."""
        import pandas as pd
        
        try:
            # Load plida4.csv
            plida4_df = pd.read_csv('resources/plida4.csv')
            
            # Filter for CORE dataset only
            core_variables = plida4_df[plida4_df['dataset_id'] == 'CORE'].copy()
            
            if core_variables.empty:
                st.warning("No variables found in CORE dataset")
                return []
            
            # Enhanced keyword matching with semantic awareness
            results = []
            query_lower = user_input.lower()
            
            # Define semantic keyword groups for better matching
            age_keywords = ['age', 'birth', 'born', 'old', 'young', 'dob', 'year_of_birth', 'month_of_birth', 'respondent', 'person']
            gender_keywords = ['gender', 'sex', 'male', 'female', 'man', 'woman', 'core_gender']
            location_keywords = ['location', 'place', 'country', 'birth_ctry', 'birthplace', 'bplp', 'born']
            death_keywords = ['death', 'died', 'deceased', 'mortality', 'year_of_death', 'month_of_death']
            marital_keywords = ['marital', 'married', 'marriage', 'spouse', 'partner', 'marital_status']
            
            # Track scoring for debugging
            scoring_debug = []
            
            # Score each CORE variable
            for _, row in core_variables.iterrows():
                description = str(row['description']).lower()
                variable_name = str(row['variable_name']).lower()
                combined_text = f"{description} {variable_name}"
                
                score = 0
                debug_reasons = []
                
                # Check for age-related queries with specific logic
                if 'age' in query_lower or 'respondent' in query_lower:
                    # Age can be derived from birth date variables
                    if any(term in combined_text for term in ['birth', 'dob', 'year_of_birth', 'month_of_birth']):
                        score += 5  # Higher score for age-birth connection
                        debug_reasons.append(f"Age-birth match (+5)")
                    if 'sex' in combined_text or 'gender' in combined_text:
                        score += 2  # Demographic variables often go together
                        debug_reasons.append(f"Demographic bonus (+2)")
                
                # Check for gender-related queries
                if any(kw in query_lower for kw in ['gender', 'sex', 'male', 'female']):
                    if any(kw in combined_text for kw in ['sex', 'gender', 'core_gender']):
                        score += 5
                        debug_reasons.append(f"Gender match (+5)")
                
                # Check for birth/location queries (be more specific)
                if any(kw in query_lower for kw in ['country', 'place', 'location', 'where']):
                    if any(kw in combined_text for kw in ['birth_ctry', 'birthplace', 'bplp', 'country']):
                        score += 4
                        debug_reasons.append(f"Location match (+4)")
                
                # Check for death-related queries
                if any(kw in query_lower for kw in death_keywords):
                    if any(kw in combined_text for kw in death_keywords):
                        score += 4
                        debug_reasons.append(f"Death match (+4)")
                
                # Check for marital status queries
                if any(kw in query_lower for kw in marital_keywords):
                    if any(kw in combined_text for kw in marital_keywords):
                        score += 4
                        debug_reasons.append(f"Marital match (+4)")
                
                # Boost common demographic variables for general queries
                if 'respondent' in query_lower or 'person' in query_lower:
                    if any(term in combined_text for term in ['sex', 'birth', 'gender']):
                        score += 2
                        debug_reasons.append(f"Respondent demographic (+2)")
                
                # General term matching (lower weight) - but skip common words that might cause noise
                query_terms = query_lower.split()
                skip_terms = {'of', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'respondent'}
                for term in query_terms:
                    if len(term) > 2 and term not in skip_terms:  # Skip very short terms and noise words
                        if term in combined_text:
                            score += 1
                            debug_reasons.append(f"Term '{term}' (+1)")
                
                # Store debug info
                if score > 0:
                    scoring_debug.append({
                        'variable': f"{variable_name} - {description}",
                        'score': score,
                        'reasons': debug_reasons
                    })
                
                if score > 0:
                    # Normalize score to 0-1 range
                    normalized_score = min(score / 10.0, 1.0)
                    
                    result = {
                        'score': normalized_score,
                        'row': {
                            'dataset_id': row['dataset_id'],
                            'dataset': row['dataset'],
                            'variable_name': row['variable_name'],
                            'description': row['description']
                        },
                        'population_match': {
                            'match': 'yes',
                            'reasoning': 'CORE dataset contains demographic variables that typically match most populations',
                            'confidence': 0.8
                        }
                    }
                    results.append(result)
            
            # Sort by score and return top results
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            top_results = results[:3]
            
            if top_results:
                st.success(f"Found {len(top_results)} relevant variables from CORE dataset (manual search)")
                # Debug info
                with st.expander("🔍 CORE Search Debug Info"):
                    st.write(f"**Query:** {user_input}")
                    st.write(f"**Total CORE variables found:** {len(core_variables)}")
                    st.write(f"**Variables with scores > 0:** {len(results)}")
                    st.write("**Top matching variables:**")
                    for i, result in enumerate(top_results):
                        st.write(f"{i+1}. **{result['row']['variable_name']}** - {result['row']['description']} (Score: {result['score']:.3f})")
                    
                    # Show some age-related CORE variables for comparison
                    st.write("**Available age-related CORE variables:**")
                    age_vars = core_variables[core_variables['description'].str.contains('birth|age|dob', case=False, na=False)]
                    for _, var in age_vars.head(5).iterrows():
                        st.write(f"- **{var['variable_name']}** - {var['description']}")
                    
                    # Show detailed scoring debug
                    st.write("**Scoring Debug (top 10):**")
                    scoring_debug_sorted = sorted(scoring_debug, key=lambda x: x['score'], reverse=True)
                    for i, debug_item in enumerate(scoring_debug_sorted[:10]):
                        st.write(f"{i+1}. {debug_item['variable']} (Score: {debug_item['score']})")
                        st.write(f"   Reasons: {', '.join(debug_item['reasons'])}")
            else:
                st.warning("No matching variables found in CORE dataset")
            
            return top_results
            
        except Exception as e:
            st.error(f"Error in manual CORE search: {e}")
            return []
    
    def _build_analysis_prompt(self, user_input):
        """Build the prompt for query analysis."""
        return (
            f"For the question: '{user_input}'\n"
            "1. Score its relevance (0-10) as a research question answerable with the ABS Person-Level Data Asset (PLIDA), where 0 is irrelevant and 10 is highly relevant.\n"
            "2. Identify the broad topic of the question: 'immigration', 'education', 'healthcare', 'poverty', 'social services', or 'unemployment'.\n"
            "3. If the score is 6 or higher, provide a list of variables to measure or construct to answer it. "
            "For demographic variables (e.g., age, sex, gender, year of birth, indigenous status, location, state), "
            "append '(demography)' to the description. "
            "For higher education variables (e.g., university degree, tertiary qualification, bachelor, master, phd, postgraduate, undergraduate, academic qualification), "
            "append '(highered)' to the description. "
            "For employment-related variables (e.g., employment status, unemployment, jobseeker, labor market, workforce, job type, occupation, retrenchment), "
            "append '(employment)' to the description. "
            "Return a valid JSON object with 'relevance_score' (int), 'topic' (string), and 'variables' (list of strings), e.g., "
            "{'relevance_score': 8, 'topic': 'unemployment', 'variables': ['Age of respondent (demography)', 'Employment status (employment)', 'University degree attainment (highered)']}."
        )
    
    def _build_medical_detection_prompt(self, user_input):
        """Build prompt to detect medical condition queries."""
        return (
            f"Analyze the following research question: '{user_input}'\n\n"
            "Determine if this question involves medical conditions, healthcare services, diseases, or health-related research.\n"
            "Look for ANY mentions of:\n"
            "- Medical conditions/diseases (e.g., diabetes, cancer, heart disease, mental health, depression, anxiety)\n"
            "- Healthcare services (e.g., hospital visits, treatments, medications, therapy)\n"
            "- Health outcomes (e.g., mortality, morbidity, recovery rates, survival, death)\n"
            "- Medical procedures or interventions\n"
            "- Health insurance or pharmaceutical benefits\n"
            "- Terms like: suffer, disease, illness, condition, health, medical, patient, treatment\n\n"
            "Be LIBERAL in detection - if there's ANY health or medical aspect, mark it as medical.\n"
            "ALWAYS suggest PBS dataset for ANY medical query, even if indirect.\n\n"
            "Return a JSON object with:\n"
            "- 'is_medical': boolean (true if ANY medical/health aspect is detected)\n"
            "- 'confidence': float (0.0 to 1.0, confidence in this assessment)\n"
            "- 'medical_keywords': list of strings (medical/health terms identified)\n"
            "- 'suggest_pbs': boolean (ALWAYS true if is_medical is true)\n"
            "- 'recommendation': string (explanation of why PBS dataset with DRG_TYPE_CD is relevant)\n\n"
            "Example: {'is_medical': true, 'confidence': 0.95, 'medical_keywords': ['cancer', 'suffer'], 'suggest_pbs': true, 'recommendation': 'PBS dataset contains DRG_TYPE_CD variable which captures diagnosis-related groups essential for analyzing cancer prevalence and related medical conditions'}"
        )
    
    def _build_geographic_detection_prompt(self, user_input):
        """Build prompt to detect geographic analysis queries."""
        return (
            f"Analyze the following research question: '{user_input}'\n\n"
            "Determine if this question involves geographic or spatial analysis.\n"
            "Look for ANY mentions of:\n"
            "- Geographic locations (e.g., states, cities, regions, countries, areas)\n"
            "- Spatial comparisons (e.g., by location, across regions, state differences)\n"
            "- Geographic terms (e.g., urban vs rural, metropolitan, remote, geographic distribution)\n"
            "- Location-based analysis (e.g., by state, by region, geographic patterns)\n"
            "- Spatial keywords: location, place, area, territory, district, zone, local, regional\n\n"
            "Be LIBERAL in detection - if there's ANY geographic or spatial aspect, mark it as geographic.\n"
            "Geographic analysis often requires location-specific datasets and variables.\n\n"
            "Return a JSON object with:\n"
            "- 'is_geographic': boolean (true if ANY geographic/spatial aspect is detected)\n"
            "- 'confidence': float (0.0 to 1.0, confidence in this assessment)\n"
            "- 'geographic_keywords': list of strings (geographic/spatial terms identified)\n"
            "- 'analysis_type': string (e.g., 'by_state', 'regional_comparison', 'urban_rural', 'location_based')\n"
            "- 'recommendation': string (explanation of geographic considerations for analysis)\n\n"
            "Example: {'is_geographic': true, 'confidence': 0.9, 'geographic_keywords': ['by state', 'cancer'], 'analysis_type': 'by_state', 'recommendation': 'Geographic analysis by state requires location-specific variables and datasets that contain state-level identifiers for proper spatial comparison'}"
        )
    
    def _build_geographic_variables_prompt(self, user_input, geographic_info, relevant_datasets):
        """Build prompt to suggest geographic variables for analysis."""
        datasets_str = ", ".join(relevant_datasets) if relevant_datasets else "available datasets"
        analysis_type = geographic_info.get('analysis_type', 'location_based')
        geographic_keywords = ", ".join(geographic_info.get('geographic_keywords', []))
        
        # Load actual variables from the relevant datasets
        variables_context = self._load_variables_for_datasets(relevant_datasets)
        
        return (
            f"Research Question: '{user_input}'\n\n"
            f"Geographic Analysis Type: {analysis_type}\n"
            f"Geographic Keywords Identified: {geographic_keywords}\n"
            f"Main Datasets for Analysis: {datasets_str}\n\n"
            f"ACTUAL VARIABLES AVAILABLE IN THESE DATASETS:\n{variables_context}\n\n"
            "Based on this geographic research question and the ACTUAL variables available in the identified datasets, "
            "select the most relevant geographic variables for this analysis.\n\n"
            "IMPORTANT: Only suggest variables that actually exist in the provided variable list above.\n\n"
            "Focus on variables that provide:\n"
            "- **Geographic Identifiers**: State, SA1, SA2, SA3, SA4, postcode, region codes\n"
            "- **Administrative Boundaries**: Political/administrative divisions\n"
            "- **Spatial Classifications**: Urban/rural, remoteness, metropolitan areas\n"
            "- **Location Attributes**: Geographic coordinates, area classifications\n\n"
            "Return a JSON object with:\n"
            "- 'selected_variables': list of exact variable names from the provided list\n"
            "- 'variable_descriptions': list of descriptions for selected variables\n"
            "- 'analysis_rationale': string explaining why these specific variables are important\n"
            "- 'dataset_sources': list of datasets containing the selected variables\n\n"
            "Example: {'selected_variables': ['STATE_11', 'SA2UCP_11', 'POSTCODE_11'], 'variable_descriptions': ['State code in 2011', 'Statistical Area Level 2 (Usual residence) in 2011', 'Postcode in 2011'], 'analysis_rationale': 'State-level cancer analysis requires STATE_11 for state grouping and SA2UCP_11 for detailed geographic areas within states', 'dataset_sources': ['Census of Population and Housing ALCD 2011-2016 5% sample']}"
        )
    
    def _load_variables_for_datasets(self, relevant_datasets):
        """Load variable descriptions for the relevant datasets."""
        try:
            # Load the variables CSV
            variables_df = pd.read_csv(VARIABLES_PATH)
            
            if not relevant_datasets:
                return "No datasets specified"
            
            # Filter variables for relevant datasets
            dataset_vars = []
            for dataset_name in relevant_datasets:
                # Try different matching strategies
                matches = variables_df[
                    (variables_df['dataset'].str.contains(dataset_name, case=False, na=False)) |
                    (variables_df['dataset_id'].str.contains(dataset_name, case=False, na=False))
                ]
                
                if matches.empty:
                    # Try partial matching for dataset names
                    for _, row in variables_df.iterrows():
                        if any(word.lower() in row['dataset'].lower() for word in dataset_name.split() if len(word) > 3):
                            matches = pd.concat([matches, row.to_frame().T])
                
                dataset_vars.append(matches)
            
            if dataset_vars:
                combined_vars = pd.concat(dataset_vars, ignore_index=True).drop_duplicates()
                
                # Focus on geographic variables
                geographic_patterns = [
                    'state', 'SA1', 'SA2', 'SA3', 'SA4', 'postcode', 'region', 'location',
                    'urban', 'rural', 'metro', 'remote', 'area', 'geographic', 'spatial',
                    'boundary', 'jurisdiction', 'district', 'zone', 'territory'
                ]
                
                geographic_vars = combined_vars[
                    combined_vars.apply(lambda row: 
                        any(pattern.lower() in row['variable_name'].lower() or 
                            pattern.lower() in row['description'].lower() 
                            for pattern in geographic_patterns), axis=1)
                ]
                
                if not geographic_vars.empty:
                    # Format variables for the prompt
                    var_list = []
                    for _, row in geographic_vars.head(50).iterrows():  # Limit to avoid token limits
                        var_list.append(f"- {row['variable_name']}: {row['description']} (Dataset: {row['dataset']})")
                    
                    return "\n".join(var_list)
                else:
                    return "No geographic variables found in the specified datasets"
            else:
                return "No variables found for the specified datasets"
                
        except Exception as e:
            return f"Error loading variables: {str(e)}"
    
    def _parse_response(self, response):
        """Parse and validate the JSON response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                raise ValueError("Empty response from LLM")
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate structure
            required_keys = {"relevance_score", "topic", "variables"}
            if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
                raise ValueError(f"Invalid JSON structure: missing required keys {required_keys}")
            
            return {
                'relevance_score': data['relevance_score'],
                'topic': data['topic'].lower(),
                'variables': data['variables'],
                'raw_response': response
            }
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing failed: {e}")
            st.write("Raw response:", response)
            raise
        except ValueError as e:
            st.error(f"Response validation failed: {e}")
            st.write("Raw response:", response)
            raise
    
    def _parse_medical_response(self, response):
        """Parse and validate the medical detection response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_medical_response()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'is_medical': data.get('is_medical', False),
                'confidence': data.get('confidence', 0.0),
                'medical_keywords': data.get('medical_keywords', []),
                'suggest_pbs': data.get('suggest_pbs', False),
                'recommendation': data.get('recommendation', ''),
                'raw_response': response
            }
            
        except Exception as e:
            st.warning(f"Error parsing medical detection response: {e}")
            return self._default_medical_response()
    
    def _default_medical_response(self):
        """Return default medical response when parsing fails."""
        return {
            'is_medical': False,
            'confidence': 0.0,
            'medical_keywords': [],
            'suggest_pbs': False,
            'recommendation': '',
            'raw_response': ''
        }
    
    def _parse_geographic_response(self, response):
        """Parse and validate the geographic detection response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_geographic_response()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'is_geographic': data.get('is_geographic', False),
                'confidence': data.get('confidence', 0.0),
                'geographic_keywords': data.get('geographic_keywords', []),
                'analysis_type': data.get('analysis_type', ''),
                'recommendation': data.get('recommendation', ''),
                'raw_response': response
            }
            
        except Exception as e:
            st.warning(f"Error parsing geographic detection response: {e}")
            return self._default_geographic_response()
    
    def _default_geographic_response(self):
        """Return default geographic response when parsing fails."""
        return {
            'is_geographic': False,
            'confidence': 0.0,
            'geographic_keywords': [],
            'analysis_type': '',
            'recommendation': '',
            'raw_response': ''
        }
    
    def _parse_geographic_variables_response(self, response):
        """Parse and validate the geographic variables suggestion response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_geographic_variables_response()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'selected_variables': data.get('selected_variables', []),
                'variable_descriptions': data.get('variable_descriptions', []),
                'analysis_rationale': data.get('analysis_rationale', ''),
                'dataset_sources': data.get('dataset_sources', []),
                'raw_response': response
            }
            
        except Exception as e:
            st.warning(f"Error parsing geographic variables response: {e}")
            return self._default_geographic_variables_response()
    
    def _default_geographic_variables_response(self):
        """Return default geographic variables response when parsing fails."""
        return {
            'selected_variables': ['STATE', 'POSTCODE', 'SA2'],
            'variable_descriptions': ['State identifier', 'Postcode', 'Statistical Area Level 2'],
            'analysis_rationale': 'Geographic analysis requires location identifiers',
            'dataset_sources': ['Census data'],
            'raw_response': ''
        }