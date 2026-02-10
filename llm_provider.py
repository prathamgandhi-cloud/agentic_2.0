from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class LLM_provider():
    def __init__(self):
        self._llm=None
        self.embedding=None
    
    def get_llm(self):
        if self._llm is None:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            deployment_name = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o")
            temperature = float(os.getenv("AZURE_TEMPERATURE", "0.1"))
            self._llm=AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                deployment_name=deployment_name,
                temperature=temperature
            )
        return self._llm
    def get_embedding_model(self):
        if self.embedding is None:
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
            api_key = os.getenv("AZURE_EMBEDDING_API_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.embedding=AzureOpenAIEmbeddings(
                model=model,
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        return self.embedding


    def embed_query(self,text):
        embedding_model=self.get_embedding_model()
        return embedding_model.embed_query(text)
    
