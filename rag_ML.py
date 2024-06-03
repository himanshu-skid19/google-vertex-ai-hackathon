import logging
import sys
import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from IPython.display import Markdown, display
from llama_index.llms.anthropic import Anthropic
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from config import pinecone_api_key,sentence_trans_hf,cohere_api_key,claude_api_key
from llama_index.core import Settings
from llama_index.core import StorageContext

#logging all information to the terminal 
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


#declaring environment variables
os.environ["HF_TOKEN"] = sentence_trans_hf

os.environ["PINECONE_API_KEY"] = pinecone_api_key
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
cohere_api_key = cohere_api_key
os.environ["COHERE_API_KEY"] = cohere_api_key
os.environ["ANTHROPIC_API_KEY"] = claude_api_key


lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
embed_model = LangchainEmbedding(lc_embed_model)
Settings.embed_model = embed_model
Settings.chunk_size = 512


async def get_ml_questions(query):
    llm = Anthropic(model="claude-3-haiku-20240307")
    Settings.llm = llm
    pinecone_index = pc.Index("mlquestions")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response
