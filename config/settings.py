import os
import numpy as np
import torch
import tomllib

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# API Configuration
def load_secrets():
    """Load secrets from secrets.toml file."""
    try:
        with open("secrets.toml", "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("secrets.toml file not found. Please create it with your OpenAI API key.")
    except Exception as e:
        raise Exception(f"Error loading secrets.toml: {e}")

secrets = load_secrets()
OPENAI_API_KEY = secrets.get("openai", {}).get("api_key")

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