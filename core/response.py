import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

@dataclass
class VariableMatch:
    """Represents a specific variable found in a dataset."""
    dataset_id: str
    variable_name: str
    description: Optional[str] = None
    relevance_score: Optional[float] = None

@dataclass 
class ConceptualVariable:
    """Represents a conceptual variable with its type and actual variable matches."""
    name: str
    variable_type: str  # causal-treatment, causal-outcome, causal-control, descriptive-main, descriptive-subgroup
    actual_variables: List[VariableMatch] = field(default_factory=list)
    recommended_datasets: List[str] = field(default_factory=list)  # Dataset IDs when no variables found

@dataclass
class Response:
    """
    Structured response object containing JSON representation of query results.
    
    This class captures the complete analysis of a user query including:
    - Query type (causal or descriptive)
    - Conceptual variables with their types and matches
    - Datasets identified as relevant
    - Internal metadata for processing
    """
    query: str
    query_type: str  # "causal" or "descriptive"
    conceptual_variables: List[ConceptualVariable] = field(default_factory=list)
    relevant_datasets: List[str] = field(default_factory=list)  # Dataset IDs from initial search
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Internal metadata (not shown to user)
    _confidence: float = 0.0
    _topic: str = ""
    _is_longitudinal: bool = False
    _relevance_score: float = 0.0
    
    def add_conceptual_variable(self, name: str, variable_type: str) -> ConceptualVariable:
        """Add a new conceptual variable and return it for further modification."""
        # Debug: Track Employment income additions
        if name == "Employment income":
            print(f"DEBUG: add_conceptual_variable called for 'Employment income' with type '{variable_type}'")
            print(f"DEBUG: Current conceptual variables: {[var.name for var in self.conceptual_variables]}")
        
        # Check if a conceptual variable with this name already exists
        for existing_var in self.conceptual_variables:
            if existing_var.name == name:
                if name == "Employment income":
                    print(f"DEBUG: Found existing 'Employment income' variable, returning it")
                return existing_var  # Return existing variable instead of creating duplicate
        
        # Create new conceptual variable if it doesn't exist
        if name == "Employment income":
            print(f"DEBUG: Creating new 'Employment income' variable")
        conceptual_var = ConceptualVariable(name=name, variable_type=variable_type)
        self.conceptual_variables.append(conceptual_var)
        return conceptual_var
    
    def add_variable_match(self, conceptual_var_name: str, dataset_id: str, variable_name: str, 
                          description: str = None, relevance_score: float = None) -> None:
        """Add a variable match to an existing conceptual variable."""
        for conceptual_var in self.conceptual_variables:
            if conceptual_var.name == conceptual_var_name:
                match = VariableMatch(
                    dataset_id=dataset_id,
                    variable_name=variable_name, 
                    description=description,
                    relevance_score=relevance_score
                )
                conceptual_var.actual_variables.append(match)
                break
    
    def add_dataset_recommendations(self, conceptual_var_name: str, dataset_ids: List[str]) -> None:
        """Add dataset recommendations when no specific variables are found."""
        for conceptual_var in self.conceptual_variables:
            if conceptual_var.name == conceptual_var_name:
                conceptual_var.recommended_datasets.extend(dataset_ids)
                break
    
    def set_metadata(self, confidence: float = None, topic: str = None, 
                    is_longitudinal: bool = None, relevance_score: float = None) -> None:
        """Set internal metadata."""
        if confidence is not None:
            self._confidence = confidence
        if topic is not None:
            self._topic = topic
        if is_longitudinal is not None:
            self._is_longitudinal = is_longitudinal
        if relevance_score is not None:
            self._relevance_score = relevance_score
    
    def to_json(self, include_metadata: bool = False) -> str:
        """
        Convert response to JSON string.
        
        Args:
            include_metadata: Whether to include internal metadata in output
        """
        data = {
            "query": self.query,
            "query_type": self.query_type,
            "conceptual_variables": [],
            "relevant_datasets": self.relevant_datasets,
            "timestamp": self.timestamp
        }
        
        # Convert conceptual variables
        for conceptual_var in self.conceptual_variables:
            # Sort actual variables by relevance score in descending order
            sorted_variables = sorted(
                conceptual_var.actual_variables,
                key=lambda x: x.relevance_score or 0,  # Handle None values
                reverse=True
            )
            
            var_data = {
                "name": conceptual_var.name,
                "variable_type": conceptual_var.variable_type,
                "actual_variables": [
                    {
                        "dataset_id": match.dataset_id,
                        "variable_name": match.variable_name,
                        "description": match.description,
                        "relevance_score": match.relevance_score
                    }
                    for match in sorted_variables
                ],
                "recommended_datasets": conceptual_var.recommended_datasets
            }
            data["conceptual_variables"].append(var_data)
        
        if include_metadata:
            data["_metadata"] = {
                "confidence": self._confidence,
                "topic": self._topic,
                "is_longitudinal": self._is_longitudinal,
                "relevance_score": self._relevance_score
            }
        
        return json.dumps(data, indent=2)
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Convert response to dictionary.
        
        Args:
            include_metadata: Whether to include internal metadata in output
        """
        return json.loads(self.to_json(include_metadata=include_metadata))
    
    @classmethod
    def from_analysis_data(cls, user_input: str, gpt_data: Dict, causal_data: Dict, 
                          categorized: Dict, dataset_results: List) -> 'Response':
        """
        Create Response object from existing analysis data.
        
        Args:
            user_input: Original user query
            gpt_data: GPT analysis results
            causal_data: Causal analysis results  
            categorized: Categorized variables
            dataset_results: Dataset search results
        """
        print(f"DEBUG: Response.from_analysis_data called")
        print(f"DEBUG: Categorized variables: {categorized}")
        # Determine query type
        is_causal = causal_data.get('is_causal', False) and causal_data.get('confidence', 0) > 0.6
        query_type = "causal" if is_causal else "descriptive"
        
        # Create response object
        response = cls(query=user_input, query_type=query_type)
        
        # Set metadata
        response.set_metadata(
            confidence=causal_data.get('confidence', 0.0),
            topic=gpt_data.get('topic', ''),
            is_longitudinal=gpt_data.get('is_longitudinal', False),
            relevance_score=gpt_data.get('relevance_score', 0.0)
        )
        
        # Add relevant datasets from initial search
        if dataset_results:
            response.relevant_datasets = [
                result['row']['dataset_id'] 
                for result in dataset_results 
                if result['row'].get('dataset_id')
            ]
        
        # Add conceptual variables based on query type
        if is_causal:
            # Add causal variables
            for var_name in categorized.get('causal_variables', []):
                response.add_conceptual_variable(var_name, "causal-treatment")
            
            for var_name in categorized.get('outcome_variables', []):
                response.add_conceptual_variable(var_name, "causal-outcome")
            
            for var_name in categorized.get('control_variables', []):
                response.add_conceptual_variable(var_name, "causal-control")
        else:
            # Add descriptive variables
            for var_name in categorized.get('main_variables', []):
                response.add_conceptual_variable(var_name, "descriptive-main")
            
            for var_name in categorized.get('subgroups', []):
                response.add_conceptual_variable(var_name, "descriptive-subgroup")
        
        return response