import os
import numpy as np
import torch
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# API Configuration
def load_openai_api_key():
    """Load OpenAI API key from various sources."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            # Try different possible key names
            api_key = (st.secrets.get("OPENAI_API_KEY") or 
                      st.secrets.get("openai_api_key") or
                      st.secrets.get("openai", {}).get("api_key"))
            if api_key:
                return api_key
    except (ImportError, AttributeError, FileNotFoundError, Exception):
        pass
    
    # Try environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Fallback to local secrets.toml file
    try:
        with open("secrets.toml", "rb") as f:
            secrets_data = tomllib.load(f)
            return secrets_data.get("openai", {}).get("api_key")
    except FileNotFoundError:
        pass
    except Exception as e:
        raise Exception(f"Error loading secrets.toml: {e}")
    
    raise ValueError(
        "OpenAI API key not found. Please set it via:\n"
        "- Streamlit Cloud: Add 'OPENAI_API_KEY' to app secrets\n"
        "- Local development: Create secrets.toml with [openai] api_key = 'your-key'\n"
        "- Environment variable: Set OPENAI_API_KEY"
    )

OPENAI_API_KEY = load_openai_api_key()

# Data Paths
DATASETS_PATH = "resources/datasets.csv"
VARIABLES_PATH = "resources/plida4.csv"

# Embedding Paths
DATASET_EMBEDDINGS_PATH = "dataset_embeddings.pkl"
VARIABLE_EMBEDDINGS_PATH = "variable_embeddings.pkl"
CORE_EMBEDDINGS_PATH = "core_embeddings.pkl"
HE_EMBEDDINGS_PATH = "he_embeddings.pkl"
NON_NDIS_EMBEDDINGS_PATH = "non_ndis_embeddings.pkl"
CENSUS_EMBEDDINGS_PATH = "census_embeddings.pkl"
NON_ACLD_EMBEDDINGS_PATH = "non_acld_embeddings.pkl"
ATO_EMBEDDINGS_PATH = "ato_embeddings.pkl"
DOMINO_EMBEDDINGS_PATH = "domino_embeddings.pkl"

# Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GPT_MODEL_NAME = "gpt-4o"

# Search Configuration
DEFAULT_TOP_K = 3
MIN_SCORE_THRESHOLD = 0.3

# UI Configuration
TEXT_STREAM_DELAY = 0.005
MAX_DESCRIPTION_LENGTH = 100