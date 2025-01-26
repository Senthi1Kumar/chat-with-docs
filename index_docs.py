from ingest.processor import DoclingProcessor
from embed.embed_chunks import MilvusManager

def main():
    print("🚀 Starting document ingestion...")
    
    # Process documents
    processor = DoclingProcessor()
    chunks = processor.process_docs()

    # Initialize and store embeddings
    milvus_mgr = MilvusManager()
    milvus_mgr.milvus_collection()
    milvus_mgr.store_embeddings(chunks)
    # test = milvus_mgr.search_embeddings("types of folders")
    # print("Test result:", test)
    # print("Collection schema:", milvus_mgr.client.describe_collection("ellie_docs"))

    records = milvus_mgr.client.query(
        collection_name="ellie_docs",
        filter="id > 0",
        output_fields=["count(*)"]
    )
    print(f"Total records in Milvus: {records[0]['count(*)']}")

    print("✅ Successfully stored embeddings!")

if __name__ == "__main__":
    main()