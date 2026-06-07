import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

def init_clients():
    """Initializes and returns the LLMod and Pinecone clients."""
    load_dotenv()
    
    llmod_key = os.environ.get("LLMOD_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    # Failsafes to catch missing keys
    if not llmod_key:
        raise ValueError("LLMOD_API_KEY is missing! Make sure it is in your .env file.")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is missing! Make sure it is in your .env file.")

    llmod_client = OpenAI(
        api_key=llmod_key,
        base_url="https://api.llmod.ai"
    )
    
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index("medium-rag") 
    
    return llmod_client, index

def embed_and_upsert(df_subset, chunk_size, overlap_ratio, chunker_function):
    """Generates embeddings via LLMod and upserts them to Pinecone in synchronized batches."""
    llmod_client, index = init_clients()
    
    # Setup Buffers to hold data
    text_batch = []
    id_batch = []
    metadata_batch = []
    
    total_chunks = 0
    BATCH_SIZE = 100

    def process_current_batch():
        """Helper function to embed and upload the current buffer, then clear it."""
        nonlocal text_batch, id_batch, metadata_batch, total_chunks
        if not text_batch:
            return
            
        try:
            # Embed the entire batch in one network request
            response = llmod_client.embeddings.create(
                input=text_batch,
                model="4UHRUIN-text-embedding-3-small"
            )
            
            # Match the returned vectors to their IDs and metadata
            vectors_to_upsert = []
            for i, data_obj in enumerate(response.data):
                embedding_vector = data_obj.embedding
                vectors_to_upsert.append((id_batch[i], embedding_vector, metadata_batch[i]))
                
            # Upsert the matched batch to Pinecone
            index.upsert(vectors=vectors_to_upsert)

            total_chunks += len(vectors_to_upsert)
            print(f"Progress: Uploaded {total_chunks} chunks so far...")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            
        # Clear the buffers to free memory
        text_batch.clear()
        id_batch.clear()
        metadata_batch.clear()

    # Main Iteration
    for idx, row in df_subset.iterrows():
        title = row.get('title', 'Unknown Title')
        text = row.get('text', '')
        url = row.get('url', '')
        authors = str(row.get('authors', 'Unknown')) 
        timestamp = str(row.get('timestamp', 'Unknown'))
        tags = str(row.get('tags', 'Unknown'))
        
        article_id = str(idx)
        chunks = chunker_function(text, chunk_size, overlap_ratio)

        for chunk_idx, chunk_content in enumerate(chunks):
            chunk_id = f"art_{article_id}_chunk_{chunk_idx}"
            text_to_embed = f"Title: {title}\nTags: {tags}\nContent: {chunk_content}"
            
            metadata = {
                "article_id": article_id,
                "title": title,
                "authors": authors,
                "timestamp": timestamp,
                "tags": tags,          
                "url": url,
                "chunk": chunk_content
            }
            
            # Add chunk data to buffers
            text_batch.append(text_to_embed)
            id_batch.append(chunk_id)
            metadata_batch.append(metadata)
            
            # Once we hit 100 items, trigger the upload process
            if len(text_batch) >= BATCH_SIZE:
                process_current_batch()
                
    # Process any leftover items that didn't cleanly divide by 100
    if text_batch:
        process_current_batch()
        
    print(f"Upload complete! Successfully processed {total_chunks} total chunks.")