import faiss
import streamlit as st
from embeddings import EmbeddingManager
from data import DataPreprocessor
from config import *

class IndexBuilder:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.embedding_manager = EmbeddingManager()
        self.indices = {}
        self.dataframes = {}
        self.embeddings = {}
    
    def build_all_indices(self):
        """Build all FAISS indices for different dataset categories."""
        # Load data
        datasets_df = self.data_loader.load_datasets()
        variables_df = self.data_loader.load_variables()
        preprocessor = DataPreprocessor(variables_df)
        
        # Build dataset index
        dataset_embeddings = self.embedding_manager.precompute_embeddings(
            datasets_df, "dataset_description", DATASET_EMBEDDINGS_PATH
        )
        d = dataset_embeddings.shape[1]
        self.indices['datasets'] = self._create_index(dataset_embeddings, d)
        self.dataframes['datasets'] = datasets_df
        self.embeddings['datasets'] = dataset_embeddings
        
        # Build variable indices
        variable_embeddings = self.embedding_manager.precompute_embeddings(
            variables_df, "description", VARIABLE_EMBEDDINGS_PATH
        )
        self.indices['variables'] = self._create_index(variable_embeddings, d)
        self.dataframes['variables'] = variables_df
        self.embeddings['variables'] = variable_embeddings
        
        # Build specialized indices
        self._build_specialized_index('core', preprocessor.get_core_dataset(), 
                                     CORE_EMBEDDINGS_PATH, d)
        self._build_specialized_index('he', preprocessor.get_he_dataset(), 
                                     HE_EMBEDDINGS_PATH, d)
        self._build_specialized_index('non_ndis', preprocessor.get_non_ndis_dataset(), 
                                     NON_NDIS_EMBEDDINGS_PATH, d)
        self._build_specialized_index('census', preprocessor.get_census_dataset(), 
                                     CENSUS_EMBEDDINGS_PATH, d)
        self._build_specialized_index('non_acld', preprocessor.get_non_acld_dataset(), 
                                     NON_ACLD_EMBEDDINGS_PATH, d)
        self._build_specialized_index('ato', preprocessor.get_ato_income_dataset(), 
                                     ATO_EMBEDDINGS_PATH, d)
        self._build_specialized_index('domino', preprocessor.get_domino_jobseeker_dataset(), 
                                     DOMINO_EMBEDDINGS_PATH, d)
        
        return self.indices, self.dataframes, self.embeddings
    
    def _create_index(self, embeddings, dimension):
        """Create a FAISS index from embeddings."""
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    def _build_specialized_index(self, name, df, embeddings_path, dimension):
        """Build a specialized index for a filtered dataset."""
        if df.empty:
            st.warning(f"No '{name}' dataset found. Using full dataset.")
            self.indices[name] = self.indices['variables']
            self.dataframes[name] = self.dataframes['variables']
            self.embeddings[name] = self.embeddings['variables']
        else:
            embeddings = self.embedding_manager.precompute_embeddings(
                df, "description", embeddings_path
            )
            self.indices[name] = self._create_index(embeddings, dimension)
            self.dataframes[name] = df
            self.embeddings[name] = embeddings