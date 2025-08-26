import json
import pandas as pd
import numpy as np
from pathlib import Path
from llm.client import LLMClient

class PopulationComparisonAnalyzer:
    """Analyzes datasets to create population comparison matrix using OpenAI."""
    
    def __init__(self):
        self.llm_client = LLMClient()
    
    def create_population_matrix(self, datasets_path, output_path):
        """Create population comparison matrix and save to CSV."""
        # Load datasets
        datasets_df = pd.read_csv(datasets_path)
        dataset_ids = datasets_df['dataset_id'].tolist()
        n_datasets = len(dataset_ids)
        
        # Initialize matrix
        comparison_matrix = np.full((n_datasets, n_datasets), -1, dtype=int)
        
        print(f"Creating population comparison matrix for {n_datasets} datasets...")
        
        # Fill diagonal with 0 (same population as itself)
        np.fill_diagonal(comparison_matrix, 0)
        
        # Compare each pair of datasets
        total_comparisons = (n_datasets * (n_datasets - 1)) // 2
        current_comparison = 0
        
        for i in range(n_datasets):
            for j in range(i + 1, n_datasets):
                current_comparison += 1
                print(f"Comparing {dataset_ids[i]} vs {dataset_ids[j]} ({current_comparison}/{total_comparisons})...")
                
                # Get dataset descriptions
                desc_i = datasets_df.iloc[i]['dataset_description']
                desc_j = datasets_df.iloc[j]['dataset_description']
                name_i = datasets_df.iloc[i]['dataset_name']
                name_j = datasets_df.iloc[j]['dataset_name']
                
                # Ask OpenAI to compare populations
                comparison_result = self._compare_dataset_populations(
                    dataset_ids[i], name_i, desc_i,
                    dataset_ids[j], name_j, desc_j
                )
                
                # Fill matrix based on result
                if comparison_result == 0:  # Same population
                    comparison_matrix[i, j] = 0
                    comparison_matrix[j, i] = 0
                elif comparison_result == 1:  # Dataset i has larger population
                    comparison_matrix[i, j] = 2  # i > j, so j < i
                    comparison_matrix[j, i] = 1  # j < i, so i > j
                elif comparison_result == 2:  # Dataset j has larger population
                    comparison_matrix[i, j] = 1  # i < j, so j > i
                    comparison_matrix[j, i] = 2  # j > i, so i < j
                else:  # Unclear
                    comparison_matrix[i, j] = 4
                    comparison_matrix[j, i] = 4
        
        # Create DataFrame with dataset IDs as index and columns
        matrix_df = pd.DataFrame(
            comparison_matrix,
            index=dataset_ids,
            columns=dataset_ids
        )
        
        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(output_path)
        
        print(f"Population comparison matrix saved to {output_path}")
        return matrix_df
    
    def _compare_dataset_populations(self, id1, name1, desc1, id2, name2, desc2):
        """Compare two datasets using OpenAI to determine population relationship."""
        prompt = self._build_comparison_prompt(id1, name1, desc1, id2, name2, desc2)
        system_prompt = "You are an expert in Australian administrative datasets and population analysis. Return only valid JSON."
        
        try:
            response = self.llm_client.complete(system_prompt, prompt)
            result = self._parse_comparison_response(response)
            return result.get('comparison', 4)  # Default to unclear
        except Exception as e:
            print(f"Error comparing {id1} vs {id2}: {e}")
            return 4  # Default to unclear on error
    
    def _build_comparison_prompt(self, id1, name1, desc1, id2, name2, desc2):
        """Build prompt for population comparison."""
        return (
            f"Compare these two Australian administrative datasets to determine their population relationship:\n\n"
            f"DATASET A:\n"
            f"ID: {id1}\n"
            f"Name: {name1}\n"
            f"Description: {desc1}\n\n"
            f"DATASET B:\n"
            f"ID: {id2}\n"
            f"Name: {name2}\n"
            f"Description: {desc2}\n\n"
            "Determine the population relationship between these datasets:\n\n"
            "Consider:\n"
            "- Population coverage (whole population vs. subsets)\n"
            "- Age restrictions (all ages vs. specific age groups)\n"
            "- Geographic scope (national vs. regional)\n"
            "- Eligibility criteria (universal vs. conditional)\n"
            "- Sample vs. full population\n\n"
            "Examples:\n"
            "- CENSUS covers entire Australian population\n"
            "- AEDC covers only children starting school (subset of population)\n"
            "- Higher Education covers only university students (subset)\n"
            "- Medicare covers most residents but has some exclusions\n"
            "- Death registrations covers all deaths (subset by definition)\n\n"
            "Return a JSON object with:\n"
            "- 'comparison': integer (0=same population, 1=A larger than B, 2=B larger than A, 4=unclear/incomparable)\n"
            "- 'rationale': string explaining the reasoning\n"
            "- 'confidence': float (0.0 to 1.0)\n\n"
            "Be conservative - if relationship is not clear, use 4 (unclear).\n\n"
            "Example: {'comparison': 1, 'rationale': 'Census covers entire Australian population while AEDC only covers children starting school, making Census a larger population', 'confidence': 0.95}"
        )
    
    def _parse_comparison_response(self, response):
        """Parse the comparison response from OpenAI."""
        try:
            # Clean response
            if response.startswith("```json"):
                response = response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif response.startswith("```"):
                response = response.split("```", 1)[1].split("```", 1)[0].strip()
            
            # Parse JSON
            data = json.loads(response)
            
            return {
                'comparison': data.get('comparison', 4),
                'rationale': data.get('rationale', ''),
                'confidence': data.get('confidence', 0.0)
            }
            
        except Exception as e:
            print(f"Error parsing comparison response: {e}")
            return {'comparison': 4, 'rationale': 'Parse error', 'confidence': 0.0}

if __name__ == "__main__":
    # Run the population analysis
    analyzer = PopulationComparisonAnalyzer()
    datasets_path = "resources/datasets.csv"
    output_path = "resources/pop_comparison.csv"
    
    try:
        matrix = analyzer.create_population_matrix(datasets_path, output_path)
        print("Population comparison matrix created successfully!")
        print("\nMatrix shape:", matrix.shape)
        print("\nSample of the matrix:")
        print(matrix.iloc[:5, :5])
    except Exception as e:
        print(f"Error creating population matrix: {e}")