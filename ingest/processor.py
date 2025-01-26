import yaml
import logging
from typing import List, Dict, Generator
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
import streamlit as st

logger = logging.getLogger(__name__)

class DoclingProcessor:
    def __init__(self, config_path: str = 'config/settings.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.chunker = HybridChunker(
            tokenizer=self.config['models']['embedding_model'],
            max_chunk_size=self.config['chunking']['chunk_size'],
            overlap=self.config['chunking']['chunk_overlap']
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def load_docs(self) -> Generator[Dict, None, None]:
        """Load and chunk web documents using Docling"""
        loader = DoclingLoader(
            file_path=self.config['web_urls'],
            export_type=ExportType.DOC_CHUNKS,
            chunker=self.chunker
            # processing_pipeline=["clean_headers", "resolve_links"]
        )
        
        try:
            for doc in loader.lazy_load():
                yield self._format_docling_chunk(doc)
        except Exception as e:
            logger.error(f'Docling loading failed: {str(e)}')
            raise

    def _format_docling_chunk(self, doc: Document) -> Dict:
        """Convert Docling format to our schema"""
        return {
            "text": doc.page_content,
            "metadata": {
                "source": doc.metadata['source'],
                "headers": doc.metadata.get('dl_meta', {}).get('headings', []),
                "docling_meta": dict(doc.metadata.get('dl_meta', {}))
            }
        }

    @st.cache_data(show_spinner=False)
    def process_docs(_self) -> List[Dict]:
        """Process documents with chunking"""
        return list(_self.load_docs())