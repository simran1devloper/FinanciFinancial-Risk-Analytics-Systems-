# milvus_handler.py
import time
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
from sentence_transformers import SentenceTransformer

MILVUS_HOST = 'milvus'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'neww_collection'
DIMENSION = 384

def connect_to_milvus(max_retries=5, retry_interval=10):
    for attempt in range(max_retries):
        try:
            connections.connect(
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            print(f"Successfully connected to Milvus on attempt {attempt + 1}")
            return
        except Exception as e:
            print(f"Failed to connect to Milvus on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
            else:
                raise

connect_to_milvus()

# Define schema for the collection
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
]

schema = CollectionSchema(fields, description="Embedding collection with text")

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    collection = Collection(name=COLLECTION_NAME)
else:
    collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": DIMENSION}
}
existing_indexes = collection.indexes
if not existing_indexes:  # No existing indexes, proceed with creating a new one
    collection.create_index(field_name="embedding", index_params=index_params)
else:
    print("Index already exists, skipping creation.")

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def insert_data(sentences, embeddings):
    embeddings_list = embeddings.tolist()
    data = [embeddings_list, sentences]
    collection.insert(data)
    collection.flush()
    collection.load()

def query_collection(query_text, top_k=10):
    query_embedding = embedding_model.encode([query_text])
    search_params = {
        "metric_type": "IP",
        "params": {"ef": 200}
    }
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    retrieved_sentences = []
    for result in results:
        for match in result:
            retrieved_sentences.append(match.entity.get("text"))
    return retrieved_sentences
