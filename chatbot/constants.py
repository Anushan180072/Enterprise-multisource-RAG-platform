import os
#Name of the admin_id
ADMIN_ID=None

#Name of the embedding model from the OpenAI
OPENAI_EMBEDDING_MODEL=None

#Name of the LLM model from the OpenAI
OPENAI_LLM_MODEL=None

#Maximum lenght of the text chunks when splitting up the large document
MAX_CHUNK_SIZE=None

#Amoount of the overlap between the consecutive text chunks
CHUNK_OVERLAP_SIZE=None

# Number of top documents to recall for initial retrieval in search operations
RECALL_TOP_K = 5

#mongodb connection string for chat histories and vectors stroting
AI_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING") 
