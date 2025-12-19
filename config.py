import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent
RAG_DATABASE_DIR = ROOT_DIR / "rag_database"
TUTORIAL_DESCRIPTIONS_DIR = RAG_DATABASE_DIR / "openfoam_tutorials"
VECTOR_STORE_DIR = RAG_DATABASE_DIR / "vector_store"

# OpenAI Configuration
# You should set these in your environment variables
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "sk-4f9c1cbdc4ce43a7886791887284e108")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = "text-embedding-v3"  # or text-embedding-ada-002

# LLM Configuration
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "sk-4f9c1cbdc4ce43a7886791887284e108")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = "qwen-plus"

# RAG Configuration
TOP_K_RESULTS = 1  # As per PRD requirement
ENABLE_VERBOSE_LOGGING = True

# Agent Configuration
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
