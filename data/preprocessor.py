import pandas as pd

class DataPreprocessor:
    def __init__(self, variables_df):
        self.variables_df = variables_df
    
    def get_core_dataset(self):
        """Filter CORE dataset."""
        return self.variables_df[self.variables_df["dataset_id"].str.upper() == "CORE"]
    
    def get_he_dataset(self):
        """Filter Higher Education dataset."""
        return self.variables_df[self.variables_df["dataset_id"].str.upper() == "HE"]
    
    def get_non_ndis_dataset(self):
        """Filter non-NDIS datasets."""
        return self.variables_df[self.variables_df["dataset_id"].str.upper() != "NDIS"]
    
    def get_census_dataset(self):
        """Filter CENSUS dataset."""
        return self.variables_df[self.variables_df["dataset_id"].str.upper() == "CENSUS"]
    
    def get_non_acld_dataset(self):
        """Filter non-ACLD datasets."""
        return self.variables_df[self.variables_df["dataset_id"].str.upper() != "ACLD"]
    
    def get_ato_income_dataset(self):
        """Filter ATO dataset for income-related variables."""
        ato_df = self.variables_df[self.variables_df["dataset_id"].str.upper() == "PIT_ITR"]
        if not ato_df.empty:
            income_df = ato_df[ato_df["description"].str.lower().str.contains("income|earnings|wage|salary")]
            if not income_df.empty:
                return income_df
        return ato_df
    
    def get_domino_jobseeker_dataset(self):
        """Filter DOMINO dataset for jobseeker-related variables."""
        domino_df = self.variables_df[self.variables_df["dataset_id"].str.upper() == "DOMINO"]
        if not domino_df.empty:
            jobseeker_df = domino_df[domino_df["description"].str.lower().str.contains("jobseeker|unemployment benefit|employment status")]
            if not jobseeker_df.empty:
                return jobseeker_df
        return domino_df