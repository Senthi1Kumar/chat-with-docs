import yaml
import logging
import tqdm
from typing import List, Dict

from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class MilvusManager:
    def __init__(self, config_path: str = 'config/settings.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config['models']['embedding_model'],
            model_kwargs={'device': self.config['device']['type']}
        )

        self.client = MilvusClient(uri=self.config['paths']['milvus_db'])
        self.milvus_collection()

    def milvus_collection(self):
        """Simplified collection setup with dynamic fields"""
        if not self.client.has_collection("ellie_docs"):
            self.client.create_collection(
                collection_name="ellie_docs",
                dimension=self.config['models']['embed_model_dim'],
                auto_id=True,
                metric_type='L2',
                index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}}
            )

    def store_embeddings(self, chunks: List[Dict]):
        """Store documents with metadata using dynamic fields"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        data = [{
            "vector": emb,
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "headers": chunk["metadata"]["headers"],
            "docling_meta": chunk["metadata"].get("docling_meta", {})
        } for emb, chunk in tqdm.tqdm(zip(embeddings, chunks), desc='Storing Embeddings')]

        self.client.insert("ellie_docs", data)
        logger.info(f'Inserted {len(data)} records')

    def search_embeddings(self, query: str):
        """Search with metadata support"""
        query_embed = self.embedder.embed_query(query)
        return self.client.search(
            collection_name="ellie_docs",
            data=[query_embed],
            limit=self.config['retrieval']['top_k'],
            output_fields=["text", "source", "headers", "docling_meta"],
            search_params={
            "metric_type": "L2",
            "params": {"nprobe": 32}}
        )