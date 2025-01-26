import os
import yaml
from typing import List

import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

class CustomMilvusRetriever(BaseRetriever, BaseModel):
    """Adapter for Docling-enhanced MilvusManager"""
    milvus_manager: object = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # print(f"Searching for: {query}")
        results = self.milvus_manager.search_embeddings(query)
        # print(f"Raw results: {results}")
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
        if isinstance(headers, list):
            return " > ".join(headers)
        return str(headers)

class GroqClient:
    def __init__(self,
                 milvus_manager,
                 config_path: str = "config/settings.yaml",
                 env_path: str = 'config/.env'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self._load_environment(env_path)
        
        self.retriever = CustomMilvusRetriever(milvus_manager=milvus_manager)
        self.llm = self._initialize_llm()
        self.qa_chain = self._create_chain()

    def _load_environment(self, env_path):
        """Load environment variables"""
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

    def _initialize_llm(self):
        """Initialize Groq LLM using streamlit secrets or .env"""
        api_key = st.secrets["GROQ_API_KEY"]
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in Streamlit secrets or in .env")
            
        return ChatGroq(
            api_key=api_key,
            model_name=self.config['groq']['llm'],
            temperature=0.3
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
            # max_tokens_limit=4000,
            # verbose=True
        )

    def generate_response(self, question: str) -> str:
        """Generate response"""
        result = self.qa_chain({"question": question})
        return self._format_response(result)
    
    def _format_response(self, result):
        """Format response with citations"""
        sources = {}
        for doc in result['source_documents']:
            domain = doc.metadata['source'].split('/')[2]
            headers = doc.metadata.get('headers', 'General Documentation')
            sources[f"{domain} ({headers})"] = doc.metadata['source']

        source_list = "\n".join(
            f"• {headers}: {source}" 
            for headers, source in sources.items()
        )
        return f"{result['answer']}\n\n**Documentation References**:\n{source_list}"
    
    def retrieve_context(self, query: str) -> List[Document]:
        """Retrieve documents only"""
        return self.retriever.get_relevant_documents(query)
