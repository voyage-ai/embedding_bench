import os
import json
import logging
import numpy as np
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from utils import parse_arguments, create_directories, Config, read_json_lines
from embedding import get_embedding_model
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embed_corpus(corpus, embedding_model, client, collection_name, batch_size=4):
    """
    Embeds the corpus and inserts the embeddings into the Milvus collection.

    Args:
        corpus (list): List of documents to be embedded.
        embedding_model: The embedding model used to generate embeddings.
        client: The Milvus client.
        collection_name (str): The name of the Milvus collection.
        batch_size (int): The size of the batch for processing embeddings.
    """
    batch_texts = []
    batch_ids = []

    for i, doc in enumerate(corpus):
        try:
            # Query to check if the document ID already exists in the collection
            results = client.get(collection_name=collection_name, ids=[doc['id']])
            if not results:
                batch_ids.append(doc['id'])
                batch_texts.append(doc['text'])

            # If the batch size is reached or it's the last document, process the batch
            if (len(batch_ids) > 0 and ((i + 1) % batch_size == 0 or (i + 1) == len(corpus))):
                # Compute embeddings for the batch
                batch_embeddings = embedding_model.forward(batch_texts)
                
                # Log the shape of the embeddings
                logger.info(f"Embedding shape: {np.array(batch_embeddings).shape}")
                
                # Prepare the records to be inserted
                records = [{"id": doc_id, "vector": embedding} for doc_id, embedding in zip(batch_ids, batch_embeddings)]
                
                # Log data types and structure
                logger.info(f"Inserting {len(records)} records into collection {collection_name}")
                for record in records:
                    logger.info(f"Record ID: {record['id']}, Vector length: {len(record['vector'])}")

                # Insert the batch into the collection
                client.insert(collection_name=collection_name, data=records)
                
                # Reset the batch lists
                batch_texts = []
                batch_ids = []

        except Exception as e:
            logger.error(f"Error processing document ID {doc['id']}: {str(e)}")

def run_queries(queries, embedding_model, client, collection_name, topk=3):
    """
    Performs queries on the Milvus collection and retrieves the top-k results.

    Args:
        queries (list): List of query documents.
        embedding_model: The embedding model used to generate query embeddings.
        client: The Milvus client.
        collection_name (str): The name of the Milvus collection.
        topk (int): The number of top results to retrieve.

    Returns:
        dict: A dictionary of query results.
    """
    retrieved_dict = {}
    for query in queries:
        try:
            query_vector = embedding_model.forward([query['text']])
            results = client.search(
                collection_name=collection_name,
                data=query_vector,
                limit=topk,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["id"],  
            )
            retrieved_dict[query['id']] = [result['id'] for result in results[0]]
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
    return retrieved_dict

def main():
    """
    Main execution function for the script.
    """
    args = parse_arguments()
    config = Config(args)
    create_directories(config)
    corpus = list(read_json_lines(f'{config.data_path}/corpus.jsonl'))
    queries = list(read_json_lines(f'{config.data_path}/queries.jsonl'))
    
    for embedding_model_name in config.embedding_models:
        try:
            model = get_embedding_model(embedding_model_name)
            client = MilvusClient(f'{config.data_path}/embedding.db')
            
            # Define the collection schema
            vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=model.dim)
            id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=20, is_primary=True, auto_id=False)
            schema = CollectionSchema(fields=[id_field, vector_field], enable_dynamic_field=True)
            
            # Prepare index parameters
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector", 
                index_type="FLAT",
                metric_type="IP",
                params={}
            )
            
            # Create collection if it doesn't exist
            collection_name = embedding_model_name.replace('-', '_')
            if not client.has_collection(collection_name=collection_name):    
                client.create_collection(
                    collection_name=collection_name,
                    dimension=model.dim,
                    schema=schema,
                    index_params=index_params,
                    metric_type="IP",  # Inner product distance
                    consistency_level="Strong",  # Strong consistency level
                )
            
            # Embed corpus and insert into Milvus collection
            embed_corpus(corpus, model, client, collection_name)
            
            # Perform queries and save results
            retrieved_dict = run_queries(queries, model, client, collection_name)
            with open(f'{config.meta_data_path}/{collection_name}.json', 'w') as f:
                f.write(json.dumps(retrieved_dict))
                
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()