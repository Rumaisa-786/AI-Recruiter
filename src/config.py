"""
Global configuration used by the ranking engine.

All constants that describe companies, locations,
AI skills and scoring live here so they are easy
to maintain.
"""

# AI Skills

AI_CORE_SKILLS = {
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "hugging face",
    "transformers",
    "bert",
    "llama",
    "mistral",
    "gemini",
    "gpt",
    "rag",
    "langchain",
    "llamaindex",
    "chromadb",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "vector database",
    "semantic search",
    "dense retrieval",
    "retrieval augmented generation",
    "sentence transformers",
    "embeddings",
    "recommendation systems",
    "information retrieval",
    "learning to rank",
    "mlflow",
    "wandb",
    "onnx",
    "fine tuning",
    "lora",
    "qlora",
}

# aliases

AI_SKILL_ALIASES = {
    "torch": "pytorch",
    "hf": "hugging face",
    "sentence embeddings": "embeddings",
    "vector db": "vector database",
    "openai api": "gpt",
    "claude": "llm",
    "langgraph": "langchain",
    "dspy": "prompt engineering",
}
NON_AI_SKILLS = {
    "react",
    "angular",
    "vue",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "java",
    "sql",
    "mongodb",
    "mysql",
    "postgresql",
}
TIER1_COMPANIES = {
    "google",
    "microsoft",
    "amazon",
    "meta",
    "apple",
    "netflix",
    "openai",
    "anthropic",
    "deepmind",
    "nvidia",
    "uber",
    "linkedin",
    "stripe",
    "flipkart",
    "phonepe",
}
CONSULTING_FIRMS = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "capgemini",
    "cognizant",
    "hcl",
}
PREFERRED_LOCATIONS = {
    "pune",
    "noida",
    "hyderabad",
    "gurugram",
    "delhi",
    "ncr",
}