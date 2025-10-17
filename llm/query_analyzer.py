import json
import pandas as pd
import streamlit as st
from .client import LLMClient
from config import VARIABLES_PATH, DATASETS_PATH

class QueryAnalyzer:
    def __init__(self):
        self.llm_client = LLMClient()
    
    def check_query_relevance(self, current_query, new_query):
        """Check if new query is relevant to current conversation."""
        prompt = self._build_relevance_prompt(current_query, new_query)
        system_prompt = "You are a helpful assistant analyzing query relevance. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_relevance_response(response)
    
    def polish_narrative_intro(self, narrative_text):
        """Polish narrative introduction for better grammar and flow."""
        prompt = self._build_narrative_polish_prompt(narrative_text)
        system_prompt = "You are a professional editor specializing in clear, engaging research communication."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            return response.strip()
        except Exception as e:
            # Return original text if polishing fails
            return narrative_text
    
    def _build_narrative_polish_prompt(self, narrative_text):
        """Build prompt for narrative polishing."""
        return (
            f"Please polish the following narrative introduction for better grammar, flow, and readability:\n\n"
            f"'{narrative_text}'\n\n"
            "Guidelines:\n"
            "- Fix any grammatical errors\n"
            "- Improve sentence flow and readability\n"
            "- Keep the same structure and meaning\n"
            "- Maintain the conversational, helpful tone\n"
            "- Keep it concise and clear\n"
            "- Preserve any markdown formatting (** for bold, * for italics)\n"
            "- Don't add extra content, just polish what's there\n\n"
            "Return only the polished text without quotes or additional commentary."
        )

    def analyze_query(self, user_input):
        """Analyze user query for relevance and extract variables."""
        prompt = self._build_analysis_prompt(user_input)
        system_prompt = "You are a helpful assistant. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        result = self._parse_response(response, user_input)
        
        # Check if query involves medical conditions
        medical_info = self.detect_medical_condition_query(user_input)
        result['medical_condition_detected'] = medical_info
        
        
        return result
    
    def detect_medical_condition_query(self, user_input):
        """Detect if query involves medical conditions and suggest PBS dataset."""
        prompt = self._build_medical_detection_prompt(user_input)
        system_prompt = "You are a healthcare data expert. Return only valid JSON."
        
        response = self.llm_client.complete(system_prompt, prompt)
        return self._parse_medical_response(response)
    
    
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
                
                # Skip population checks for CORE dataset variables
                if dataset_id == 'CORE':
                    result['population_match'] = {
                        'match': 'yes',
                        'reasoning': 'CORE dataset contains demographic variables that typically match most populations',
                        'confidence': 1.0
                    }
                    filtered_results.append(result)
                    continue
                
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
                should_show_core_alternatives = True
            else:
                # Check if any results have population warnings
                has_warnings = any(
                    result.get('population_match', {}).get('match') in ['maybe', 'error'] 
                    for result in filtered_results
                )
                if has_warnings:
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
            # Force manual search for better matching
            return self._fallback_manual_core_search(user_input)
            
            # Try to use the existing semantic search engine to search CORE variables
            # First, let's see if we can use the plida4 index if it exists
            if 'plida4' in search_engine.indices:
                # Use semantic search on plida4 index with CORE filtering
                all_results = search_engine.search_variables(
                    user_input, 
                    'plida4', 
                    top_k=50,
                    use_openai_relevance=True,
                    conceptual_variable=user_input
                )
                
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
                
                # Return results without status messages
                
                return top_results
            else:
                # Fallback to manual text matching if plida4 index doesn't exist
                return self._fallback_manual_core_search(user_input)
                
        except Exception as e:
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
            # Use variable_name as secondary sort key for deterministic ordering in case of ties
            results = sorted(results, key=lambda x: (x['score'], x['row'].get('variable_name', '')), reverse=True)
            top_results = results[:3]
            
            # Return results directly without status messages
            
            return top_results
            
        except Exception as e:
            return []
    
    def _build_analysis_prompt(self, user_input):
        """Build the prompt for query analysis."""
        return (
            f"For the question: '{user_input}'\n"
            "1. **First, reformulate the question** to make it clearer and more precise while staying faithful to the original intent. Do NOT add new concepts, variables, or analysis dimensions that were not explicitly mentioned in the original question. Do NOT add demographic breakdowns or subgroup analysis unless specifically requested. IMPORTANT: Only reformulate the single question - do NOT add additional questions or sub-questions.\n"
            "2. Score its relevance (0-10) as a research question answerable with the ABS Person-Level Data Asset (PLIDA), where 0 is irrelevant and 10 is highly relevant.\n"
            "3. Identify the broad topic of the question: 'immigration', 'education', 'healthcare', 'poverty', 'social services', or 'unemployment'.\n"
            "4. If the score is 6 or higher, provide a list of variables to measure or construct to answer it. "
            "IMPORTANT: When describing variables, follow these language guidelines: "
            "- NEVER use the word 'status' in variable descriptions (use 'employment' instead of 'employment status', 'marital information' instead of 'marital status', etc.) "
            "- NEVER use hierarchical terms like 'household head', 'head of household', 'family head' or similar. Instead use inclusive language like 'adults in household', 'primary earner', 'main income provider', or 'household members'. "
            "- Use person-first, inclusive language that doesn't assume family hierarchies. "
            "- DO NOT include indigenous information or variables UNLESS the user query explicitly mentions indigenous people, Aboriginal people, Torres Strait Islander people, First Nations, or asks specifically about indigenous topics. When the user DOES explicitly mention these groups, then DO include indigenous identity variables as they are essential for the analysis and ALWAYS append '(indig)' to ANY variable description that relates to Indigenous/Aboriginal/Torres Strait Islander identity or characteristics. For example: 'Indigenous identity (indig)', 'Aboriginal identity (indig)', 'Torres Strait Islander identity (indig)'. "
            "- For variables related to a person's migration status or visa status (including refugees, international students, skilled migrants, humanitarian visas, temporary visas, permanent residents, citizenship status, country of birth for migration context, visa types, visa classes, visa subclasses), append '(visa)' to the description. Note that Indigenous/Aboriginal status is by itself NOT related to migration/visa and should not get the (visa) suffix. "
            "- ALWAYS include location or geographic variables when the user query explicitly mentions them. For example, if the query mentions 'state', 'geography', 'location', 'area', 'region', 'suburb', 'postcode', or any geographic terms, include the relevant geographic variable. Common geographic variables include 'state', 'region', 'area', 'location'. DO NOT include geographic variables only when the query has no geographic component at all. "
            "For demographic variables (e.g., age, sex, gender, year of birth), "
            "append '(demography)' to the description. "
            "For higher education variables (e.g., university degree, tertiary qualification, bachelor, master, phd, postgraduate, undergraduate, academic qualification), "
            "append '(highered)' to the description. "
            "For employment-related variables (e.g., employment, unemployment, jobseeker, labor market, workforce, job type, retrenchment), "
            "append '(employment)' to the description. "
            "For occupation-specific variables (e.g., occupation type, occupational category, job title, profession, occupational classification, work role, job category, what occupation, which occupation), "
            "append '(occup)' to the description. "
            "5. Determine if this research question requires longitudinal data (tracking the same individuals over time). Answer 'yes' if the research requires observing changes in the same people across multiple time periods, transitions, or development over time. Answer 'no' if it's a cross-sectional analysis comparing different groups or describing population characteristics at a single point in time. "
            "Return a valid JSON object with 'reformulated_question' (string), 'relevance_score' (int), 'topic' (string), 'variables' (list of strings), and 'is_longitudinal' (boolean). "
            "Examples: "
            "{'reformulated_question': 'What is the relationship between university degree attainment and unemployment rates among different age groups in Australia?', 'relevance_score': 8, 'topic': 'unemployment', 'variables': ['Age of respondent (demography)', 'Employment (employment)', 'University degree attainment (highered)'], 'is_longitudinal': false} "
            "{'reformulated_question': 'How do poverty rates vary across different states in Australia?', 'relevance_score': 9, 'topic': 'poverty', 'variables': ['Income level', 'State', 'Age of respondent (demography)', 'Employment (employment)', 'Educational attainment'], 'is_longitudinal': false} "
            "{'reformulated_question': 'What are the employment outcomes for refugees in Australia?', 'relevance_score': 8, 'topic': 'immigration', 'variables': ['Employment (employment)', 'Refugee information', 'Immigration details'], 'is_longitudinal': false} "
            "{'reformulated_question': 'How do employment outcomes change for individuals after completing education?', 'relevance_score': 9, 'topic': 'education', 'variables': ['Employment (employment)', 'Educational attainment (highered)', 'Time periods'], 'is_longitudinal': true}"
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
    
    
    def _parse_response(self, response, original_question=None):
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
                'reformulated_question': data.get('reformulated_question', original_question or ""),
                'relevance_score': data['relevance_score'],
                'topic': data['topic'].lower(),
                'variables': data['variables'],
                'is_longitudinal': data.get('is_longitudinal', False),
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
    
    
    def _build_relevance_prompt(self, current_query, new_query):
        """Build prompt to check if new query is relevant to current conversation."""
        return (
            f"Current conversation query: '{current_query}'\n"
            f"New user input: '{new_query}'\n\n"
            "Analyze whether the new user input is relevant to the current conversation topic.\n"
            "The new input should be considered relevant if:\n"
            "- It provides additional information, clarification, or context for the current query\n"
            "- It asks follow-up questions related to the current topic\n"
            "- It refines or expands on the current research question\n"
            "- It provides constraints or additional parameters for the current analysis\n\n"
            "The new input should be considered NOT relevant if:\n"
            "- It asks about a completely different research topic\n"
            "- It starts a new, unrelated research question\n"
            "- It changes the fundamental scope of the analysis\n\n"
            "Return a JSON object with:\n"
            "- 'is_relevant': boolean (true if relevant to current conversation)\n"
            "- 'confidence': float (0.0 to 1.0, confidence in this assessment)\n"
            "- 'reasoning': string explaining the assessment\n"
            "- 'combined_query': string (if relevant, combine both queries into a single coherent question)\n\n"
            "Example: {'is_relevant': true, 'confidence': 0.9, 'reasoning': 'New input provides age constraints for the existing healthcare question', 'combined_query': 'Impact of NDIS on mental healthcare utilization among adults aged 18-65'}"
        )
    
    def _parse_relevance_response(self, response):
        """Parse and validate the relevance response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_relevance_response()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'is_relevant': data.get('is_relevant', False),
                'confidence': float(data.get('confidence', 0.0)),
                'reasoning': data.get('reasoning', 'No reasoning provided'),
                'combined_query': data.get('combined_query', ''),
                'raw_response': response
            }
            
        except Exception as e:
            st.warning(f"Error parsing relevance response: {e}")
            return self._default_relevance_response()
    
    def _default_relevance_response(self):
        """Return default relevance response when parsing fails."""
        return {
            'is_relevant': False,
            'confidence': 0.0,
            'reasoning': 'Unable to assess relevance',
            'combined_query': '',
            'raw_response': ''
        }
    
    def recommend_datasets_for_conceptual_variable(self, conceptual_variable, available_datasets):
        """Recommend 1-2 most relevant datasets for a conceptual variable when no specific variables are found."""
        prompt = self._build_dataset_recommendation_prompt(conceptual_variable, available_datasets)
        system_prompt = "You are a data expert specializing in Australian administrative datasets. Return only valid JSON."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            return self._parse_dataset_recommendation_response(response)
        except Exception as e:
            return {
                'recommended_datasets': [],
                'reasoning': f'Error getting recommendations: {str(e)}'
            }
    
    def _build_dataset_recommendation_prompt(self, conceptual_variable, available_datasets):
        """Build prompt for dataset recommendation."""
        datasets_text = "\n".join([f"- **{ds.get('dataset_id', 'Unknown')}**: {ds.get('dataset_name', 'Unknown')} - {ds.get('dataset_description', 'No description')}" for ds in available_datasets])
        
        return (
            f"Conceptual Variable Needed: '{conceptual_variable}'\n\n"
            f"Available Datasets:\n{datasets_text}\n\n"
            "Based on the conceptual variable needed and the available datasets, recommend 1-2 datasets that are most likely to contain variables related to this concept.\n\n"
            "Consider:\n"
            "- Which datasets would logically contain information about this concept\n"
            "- The scope and coverage of each dataset\n"
            "- The typical variables that would be found in administrative datasets\n"
            "- Dataset descriptions and their relevance to the conceptual variable\n\n"
            "Return a JSON object with:\n"
            "- 'recommended_datasets': list of 1-2 dataset_ids that are most relevant\n"
            "- 'reasoning': string explaining why these datasets were selected\n\n"
            "Example: {'recommended_datasets': ['CENSUS', 'ACLD'], 'reasoning': 'Census datasets typically contain demographic variables like age, income, and education that would be relevant for this analysis'}"
        )
    
    def _parse_dataset_recommendation_response(self, response):
        """Parse and validate the dataset recommendation response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            if not response:
                return self._default_dataset_recommendation_response()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'recommended_datasets': data.get('recommended_datasets', []),
                'reasoning': data.get('reasoning', 'No reasoning provided'),
                'raw_response': response
            }
            
        except Exception as e:
            return self._default_dataset_recommendation_response()
    
    def _default_dataset_recommendation_response(self):
        """Return default dataset recommendation response when parsing fails."""
        return {
            'recommended_datasets': [],
            'reasoning': 'Unable to provide dataset recommendations',
            'raw_response': ''
        }
    
    def score_dataset_relevance(self, user_query, dataset_id, dataset_name, dataset_description):
        """Score how relevant a dataset is to the user query (0.0 to 1.0)."""
        prompt = self._build_dataset_relevance_prompt(user_query, dataset_id, dataset_name, dataset_description)
        system_prompt = "You are an expert in Australian administrative datasets and research methodology. Return only a single number between 0.0 and 1.0."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            # Extract the score from the response
            score_str = response.strip().replace(',', '.')  # Handle comma decimal separators
            score = float(score_str)
            # Ensure score is within bounds
            return max(0.0, min(1.0, score))
        except Exception as e:
            # Return moderate score if scoring fails
            return 0.5
    
    def _build_dataset_relevance_prompt(self, user_query, dataset_id, dataset_name, dataset_description):
        """Build prompt for dataset relevance scoring."""
        return (
            f"User Research Query: '{user_query}'\n\n"
            f"Dataset to Evaluate:\n"
            f"- ID: {dataset_id}\n"
            f"- Name: {dataset_name}\n"
            f"- Description: {dataset_description}\n\n"
            "Score how relevant this dataset is to answering the user's research question on a scale of 0.0 to 1.0:\n"
            "- 0.0 = Completely irrelevant (no connection to the research question)\n"
            "- 0.3 = Low relevance (weak or indirect connection)\n"
            "- 0.5 = Moderate relevance (some useful information but not ideal)\n"
            "- 0.7 = High relevance (contains important data for the research)\n"
            "- 1.0 = Perfect relevance (ideal dataset for this research question)\n\n"
            "Consider:\n"
            "- Does the dataset contain variables directly related to the research question?\n"
            "- Does the population coverage match the research needs?\n"
            "- Are there data quality or completeness concerns?\n"
            "- How well does the dataset's purpose align with the research goals?\n\n"
            "Special considerations:\n"
            "- For Indigenous/Aboriginal research: COMBINED dataset should score higher than migration datasets\n"
            "- For life expectancy research: DEATHS dataset should score very high\n"
            "- For medical conditions: PBS/MBS datasets should score high\n"
            "- Migration datasets (SDB, MT_DEMOGS) should score low for Indigenous research\n"
            "- For VET/vocational education/apprenticeship research: TVA and A&T datasets should score much higher than migration datasets\n"
            "- For education and training outcomes: TVA (Total VET Activity) and A&T (Apprentice and Trainee) are primary sources, not migration datasets\n"
            "- Migration datasets (SDB, VISA, MT_DEMOGS) contain occupation data but are not relevant for VET graduate or training outcome research\n\n"
            "Return only the numerical score (e.g., 0.8):"
        )

    def generate_concise_description(self, description, context_type="dataset"):
        """Generate a concise, legible one-sentence description using OpenAI."""
        if len(description) <= 100:
            return description  # Already short enough
        
        prompt = self._build_description_prompt(description, context_type)
        system_prompt = "You are a helpful assistant that creates clear, concise descriptions. Return only the single sentence description without quotes or additional text."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            # Clean the response
            cleaned_response = response.strip().strip('"').strip("'")
            # Ensure it ends with a period
            if not cleaned_response.endswith('.'):
                cleaned_response += '.'
            return cleaned_response
        except Exception as e:
            # Fallback to truncation if OpenAI fails
            return description[:97] + "..." if len(description) > 100 else description
    
    def _build_description_prompt(self, description, context_type):
        """Build prompt for description summarization."""
        return (
            f"Convert this {context_type} description into a clear, concise single sentence that captures the key information:\n\n"
            f"Original: {description}\n\n"
            f"Requirements:\n"
            f"- Must be exactly one sentence\n"
            f"- Maximum 100 characters\n"
            f"- Keep the most important information\n"
            f"- Use clear, simple language\n"
            f"- End with a period\n"
            f"- Do not use quotes or additional formatting\n\n"
            f"Concise description:"
        )