import os
import time
import yaml
from typing import List

import streamlit as st
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

class RateLimiter:
    """Basic rate limiter for Google GenAI API"""
    def __init__(self):
        self.last_call = 0
        self.request_count = 0
        self.token_count = 0
        self.RPM_LIMIT = 15
        self.TPM_LIMIT = 1000000  # 1 million tokens
        self.RPD_LIMIT = 1500

    def check_limit(self, tokens):
        now = time.time()
        elapsed = now - self.last_call
        
        # Reset daily counters
        if elapsed > 86400:  # 24 hours
            self.request_count = 0
            self.token_count = 0
            
        if self.request_count >= self.RPD_LIMIT:
            raise Exception("Daily request limit exceeded")
            
        if self.token_count + tokens > self.TPM_LIMIT:
            raise Exception("Token limit exceeded")
            
        # Enforce RPM limit
        if self.request_count % self.RPM_LIMIT == 0 and elapsed < 60:
            sleep_time = 60 - elapsed
            time.sleep(sleep_time)

        self.last_call = time.time()
        self.request_count += 1
        self.token_count += tokens

class CustomMilvusRetriever(BaseRetriever, BaseModel):
    """Adapter for Docling-enhanced MilvusManager"""
    milvus_manager: object = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.milvus_manager.search_embeddings(query)
        return [
            Document(
                page_content=res["entity"]["text"],
                metadata={
                    "source": res["entity"]["source"],
                    "headers": self._format_headers(res["entity"]["headers"]),
                    "docling_meta": res["entity"].get("docling_meta", {})
                }
            ) for res in results[0]
        ]

    def _format_headers(self, headers) -> str:
        return " > ".join(headers) if isinstance(headers, list) else str(headers)

class GoogleGenAIClient:
    def __init__(self,
                 milvus_manager,
                 config_path: str = "config/settings.yaml",
                 env_path: str = 'config/.env'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self._load_environment(env_path)
        self.rate_limiter = RateLimiter()
        
        self.retriever = CustomMilvusRetriever(milvus_manager=milvus_manager)
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_chain()

    def _load_environment(self, env_path):
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def _initialize_llm(self):
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets or environment")
            
        return ChatGoogleGenerativeAI(
            model=self.config['google_genai']['model'],
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=5000
        )

    def _create_chain(self):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical assistant for Ellie.ai documentation. 
             Use ONLY the structured context below. Follow these rules:
             1. Leverage header hierarchy for accurate answers
             2. Reference document structure from metadata
             3. Cite sources using document headings
             4. If unsure, state "Based on Ellie's documentation: [answer]"
             
             Context: {context}"""),
            ("human", "{question}")
        ])

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": prompt_template},
            return_source_documents=True,
            memory=ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key="answer"
            ),
            max_tokens_limit=4000
        )

    def generate_response(self, question: str) -> str:
        """Generate response with rate limiting"""
        try:
            est_tokens = len(question.split()) * 2 + 500  
            self.rate_limiter.check_limit(est_tokens)
            
            result = self.qa_chain({"question": question})
            return self._format_response(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_response(self, result):
        sources = {}
        for doc in result['source_documents']:
            domain = doc.metadata['source'].split('/')[2]
            headers = doc.metadata.get('headers', 'General Documentation')
            sources[f"{domain} ({headers})"] = doc.metadata['source']

        source_list = "\n".join(
            f"• {headers}: {source}" 
            for headers, source in sources.items()
        )
        return f"{result['answer']}\n\n**References**:\n{source_list}"
    
    def retrieve_context(self, query: str) -> List[Document]:
        return self.retriever.get_relevant_documents(query)

