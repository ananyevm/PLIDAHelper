"""
Script to run population comparison analysis in smaller batches.
"""
import pandas as pd
from utils.population_analyzer import PopulationComparisonAnalyzer

def run_sample_analysis():
    """Run analysis on a smaller sample first."""
    # Read datasets
    datasets_df = pd.read_csv("resources/datasets.csv")
    
    # Use all datasets instead of a limited sample
    sample_df = datasets_df.copy()
    
    print(f"Running analysis on {len(sample_df)} sample datasets:")
    for _, row in sample_df.iterrows():
        print(f"- {row['dataset_id']}: {row['dataset_name']}")
    
    # Create analyzer
    analyzer = PopulationComparisonAnalyzer()
    
    # Save sample to temporary file
    sample_path = "resources/sample_datasets.csv"
    sample_df.to_csv(sample_path, index=False)
    
    # Run analysis
    try:
        matrix = analyzer.create_population_matrix(sample_path, "resources/sample_pop_comparison.csv")
        print("\nSample analysis completed successfully!")
        print("\nResulting matrix:")
        print(matrix)
        
        # Clean up
        import os
        os.remove(sample_path)
        
        return True
        
    except Exception as e:
        print(f"Error in sample analysis: {e}")
        return False

if __name__ == "__main__":
    success = run_sample_analysis()
    if success:
        print("\n✅ Sample analysis successful! Ready to run full analysis if needed.")
    else:
        print("\n❌ Sample analysis failed.")