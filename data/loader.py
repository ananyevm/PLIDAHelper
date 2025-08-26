import pandas as pd
import streamlit as st

class DataLoader:
    def __init__(self, datasets_path, variables_path):
        self.datasets_path = datasets_path
        self.variables_path = variables_path
        self.datasets_df = None
        self.variables_df = None
    
    def load_datasets(self):
        """Load and validate datasets CSV."""
        try:
            self.datasets_df = pd.read_csv(self.datasets_path)
            required_cols = {"dataset_id", "dataset_name", "dataset_description"}
            if not required_cols.issubset(self.datasets_df.columns):
                raise ValueError(f"datasets.csv must contain {required_cols} columns.")
            self.datasets_df["dataset_description"] = self.datasets_df["dataset_description"].fillna("").astype(str)
            return self.datasets_df
        except Exception as e:
            st.error(f"Error loading datasets: {e}")
            raise
    
    def load_variables(self):
        """Load and validate variables CSV."""
        try:
            self.variables_df = pd.read_csv(self.variables_path)
            required_cols = {"dataset_id", "dataset", "variable_name", "description"}
            if not required_cols.issubset(self.variables_df.columns):
                raise ValueError(f"variables.csv must contain {required_cols} columns.")
            self.variables_df["description"] = self.variables_df["description"].fillna("").astype(str)
            return self.variables_df
        except Exception as e:
            st.error(f"Error loading variables: {e}")
            raise
    
    def get_filtered_dataset(self, dataset_id):
        """Get variables filtered by dataset ID."""
        if self.variables_df is None:
            self.load_variables()
        return self.variables_df[self.variables_df["dataset_id"].str.upper() == dataset_id.upper()]