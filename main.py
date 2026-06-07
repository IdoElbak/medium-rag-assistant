from data_processor import load_and_subset_data, chunk_text
from vector_db import embed_and_upsert

def main():
    # --- RAG Hyperparameters ---
    FILEPATH = 'medium-english-50mb.csv'
    CHUNK_SIZE = 512
    OVERLAP_RATIO = 0.2
    
    TESTING_ROW_LIMIT = None
    
    # Process Data
    df = load_and_subset_data(FILEPATH, TESTING_ROW_LIMIT)

    # Embed and Upload
    embed_and_upsert(df, CHUNK_SIZE, OVERLAP_RATIO, chunk_text)

if __name__ == "__main__":
    main()