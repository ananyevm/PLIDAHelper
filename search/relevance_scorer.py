"""
OpenAI-based relevance scoring for semantic search results.
Evaluates how well each variable matches the conceptual variable needed.
"""

import json
from typing import Dict, List, Any, Tuple
from config import OPENAI_API_KEY

class RelevanceScorer:
    """Uses OpenAI to score relevance of variables to conceptual queries."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        if not self.llm_client and OPENAI_API_KEY:
            from llm.client import LLMClient
            self.llm_client = LLMClient()
    
    def score_variable_relevance(self, conceptual_variable: str, actual_variable: Dict[str, Any]) -> Tuple[float, str]:
        """
        Score how well an actual variable matches a conceptual variable need.
        
        Args:
            conceptual_variable: The conceptual variable description (e.g., "Mental healthcare service usage")
            actual_variable: Dict containing 'variable_name', 'description', 'dataset' etc.
            
        Returns:
            Tuple of (probability_score, explanation)
        """
        if not self.llm_client:
            return 0.5, "LLM client not available"
        
        prompt = self._build_relevance_prompt(conceptual_variable, actual_variable)
        system_prompt = "You are an expert in research data analysis. Evaluate variable relevance accurately and return only valid JSON."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            return self._parse_relevance_response(response)
        except Exception as e:
            return 0.5, f"Error evaluating relevance: {str(e)}"
    
    def score_batch_relevance(self, conceptual_variable: str, actual_variables: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
        """
        Score multiple variables for efficiency.
        
        Args:
            conceptual_variable: The conceptual variable description
            actual_variables: List of variable dictionaries
            
        Returns:
            List of (probability_score, explanation) tuples
        """
        if not self.llm_client:
            return [(0.5, "LLM client not available")] * len(actual_variables)
        
        # For batch processing, we'll process in chunks to avoid token limits
        chunk_size = 5
        all_scores = []
        
        for i in range(0, len(actual_variables), chunk_size):
            chunk = actual_variables[i:i + chunk_size]
            chunk_scores = self._score_chunk(conceptual_variable, chunk)
            all_scores.extend(chunk_scores)
        
        return all_scores
    
    def _score_chunk(self, conceptual_variable: str, variables_chunk: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
        """Score a chunk of variables together."""
        prompt = self._build_batch_relevance_prompt(conceptual_variable, variables_chunk)
        system_prompt = "You are an expert in research data analysis. Evaluate variable relevance accurately and return only valid JSON."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            return self._parse_batch_relevance_response(response, len(variables_chunk))
        except Exception as e:
            return [(0.5, f"Error: {str(e)}")] * len(variables_chunk)
    
    def _build_relevance_prompt(self, conceptual_variable: str, actual_variable: Dict[str, Any]) -> str:
        """Build prompt for single variable relevance scoring."""
        var_name = actual_variable.get('variable_name', 'Unknown')
        var_desc = actual_variable.get('description', 'No description')
        dataset = actual_variable.get('dataset', 'Unknown dataset')
        
        return f"""
Evaluate how well this actual variable matches the conceptual variable needed for research.

CONCEPTUAL VARIABLE NEEDED: "{conceptual_variable}"

ACTUAL VARIABLE TO EVALUATE:
- Variable Name: {var_name}
- Description: {var_desc}
- Dataset: {dataset}

EVALUATION CRITERIA:
1. Direct relevance: Does this variable directly measure what the conceptual variable describes?
2. Indirect relevance: Could this variable be used as a proxy or indicator?
3. Semantic similarity: Do the concepts overlap meaningfully?
4. Research utility: Would this variable be useful for studying the conceptual variable?

Provide a probability score (0.0 to 1.0) where:
- 1.0 = Perfect match, directly measures the conceptual variable
- 0.8-0.9 = Very high relevance, strong indicator or close proxy
- 0.6-0.7 = Moderate relevance, could be useful with some interpretation
- 0.4-0.5 = Low relevance, weak connection or indirect relationship
- 0.0-0.3 = Very low/no relevance, doesn't relate to the conceptual variable

Return a JSON object with:
- "probability": float (0.0 to 1.0)
- "explanation": string (brief explanation of the score)

Example: {{"probability": 0.85, "explanation": "Variable directly measures healthcare service usage, highly relevant to mental healthcare service research"}}
"""
    
    def _build_batch_relevance_prompt(self, conceptual_variable: str, variables_chunk: List[Dict[str, Any]]) -> str:
        """Build prompt for batch variable relevance scoring."""
        variables_text = ""
        for i, var in enumerate(variables_chunk):
            var_name = var.get('variable_name', 'Unknown')
            var_desc = var.get('description', 'No description')
            dataset = var.get('dataset', 'Unknown dataset')
            variables_text += f"""
Variable {i+1}:
- Name: {var_name}
- Description: {var_desc}
- Dataset: {dataset}
"""
        
        return f"""
Evaluate how well each of these actual variables matches the conceptual variable needed for research.

CONCEPTUAL VARIABLE NEEDED: "{conceptual_variable}"

ACTUAL VARIABLES TO EVALUATE:
{variables_text}

For each variable, provide a probability score (0.0 to 1.0) where:
- 1.0 = Perfect match, directly measures the conceptual variable
- 0.8-0.9 = Very high relevance, strong indicator or close proxy
- 0.6-0.7 = Moderate relevance, could be useful with some interpretation
- 0.4-0.5 = Low relevance, weak connection or indirect relationship
- 0.0-0.3 = Very low/no relevance, doesn't relate to the conceptual variable

Return a JSON object with:
- "scores": array of objects, each with "probability" (float) and "explanation" (string)

Example: {{
  "scores": [
    {{"probability": 0.85, "explanation": "Direct healthcare usage measure"}},
    {{"probability": 0.3, "explanation": "Demographic variable, not directly related to healthcare usage"}}
  ]
}}
"""
    
    def _parse_relevance_response(self, response: str) -> Tuple[float, str]:
        """Parse single variable relevance response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response)
            
            probability = float(data.get('probability', 0.5))
            explanation = str(data.get('explanation', 'No explanation provided'))
            
            # Ensure probability is in valid range
            probability = max(0.0, min(1.0, probability))
            
            return probability, explanation
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return 0.5, f"Error parsing response: {str(e)}"
    
    def _parse_batch_relevance_response(self, response: str, expected_count: int) -> List[Tuple[float, str]]:
        """Parse batch variable relevance response."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            data = json.loads(response)
            scores_data = data.get('scores', [])
            
            results = []
            for i in range(expected_count):
                if i < len(scores_data):
                    score_obj = scores_data[i]
                    probability = float(score_obj.get('probability', 0.5))
                    explanation = str(score_obj.get('explanation', 'No explanation provided'))
                    
                    # Ensure probability is in valid range
                    probability = max(0.0, min(1.0, probability))
                    results.append((probability, explanation))
                else:
                    results.append((0.5, "Missing score in response"))
            
            return results
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return [(0.5, f"Error parsing response: {str(e)}")] * expected_count
    
    def format_relevance_display(self, probability: float, explanation: str) -> str:
        """Format relevance score for display."""
        percentage = int(probability * 100)
        
        if probability >= 0.8:
            emoji = "🟢"
            level = "High"
        elif probability >= 0.6:
            emoji = "🟡"
            level = "Moderate"
        elif probability >= 0.4:
            emoji = "🟠"
            level = "Low"
        else:
            emoji = "🔴"
            level = "Very Low"
        
        return f"{emoji} {percentage}% relevance ({level}): {explanation}"