import pandas as pd

def load_and_subset_data(filepath, row_limit=None):
    """Reads the CSV. If row_limit is provided, extracts a subset. Otherwise, loads all data."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Only subset if a limit was explicitly provided
    if row_limit is not None:
        df = df.head(row_limit)
        
    print(f"Processing {len(df)} articles...")
    return df

def chunk_text(text, chunk_size, overlap_ratio):
    """Splits text into overlapping chunks based on word count."""
    if pd.isna(text) or not isinstance(text, str):
        return []

    words = text.split()
    overlap_size = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap_size
    
    # Failsafe to prevent infinite loops
    if step_size <= 0:
        step_size = chunk_size
        
    chunks = []
    for i in range(0, len(words), step_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks