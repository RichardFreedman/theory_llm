# TMI Parser ChromaDB Fix Instructions

## The Problems

1. **Readonly Database Error**: ChromaDB database is locked or corrupted
2. **Token Limit Error**: Trying to send too many chunks to OpenAI at once (zarins58.html has 1027 chunks, exceeds 300k token limit)

## The Solution

Follow these steps in your Jupyter notebook:

### Step 1: Restart the Kernel

In Jupyter, click **Kernel → Restart Kernel**. This will close all database connections.

### Step 2: Re-run Setup Cells

Run these cells in order:
1. Import libraries cell
2. API key setup cell

### Step 3: Fix Database Initialization

**Replace** the database configuration cell with this code:

```python
import shutil

# Configuration for database schema and settings
DB_CONFIG = {
    "version": "1.0",
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 2000,
    "chunk_overlap": 300,
    "collection_name": "HTML_samples_italian"
}

db_path = Path('./chroma-db_italian')
config_path = db_path / 'db_config.json'

# Always start fresh to avoid readonly errors
if db_path.exists():
    shutil.rmtree(db_path)
    print(f"✓ Deleted existing database at {db_path}")

# Create directory
db_path.mkdir(exist_ok=True)

# Save current configuration
with open(config_path, 'w') as f:
    json.dump(DB_CONFIG, f, indent=2)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=DB_CONFIG['embedding_model'])

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=DB_CONFIG['collection_name'],
    embedding_function=embeddings,
    persist_directory=str(db_path)
)

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=DB_CONFIG['chunk_size'],
    chunk_overlap=DB_CONFIG['chunk_overlap'],
    length_function=len,
    is_separator_regex=False
)

print("✓ Database initialized successfully")
```

### Step 4: Update process_html_files Function

**Replace** the `process_html_files` function (in the cell after the helper functions) with the version from `tmi_process_html_fix.py`.

The key change is adding batch processing:

```python
def process_html_files(html_dir='italian_sources', force_reprocess=False, batch_size=100):
    # ... (use the complete function from tmi_process_html_fix.py)
```

**Key addition** - this section processes chunks in batches:

```python
# Process chunks in batches to avoid OpenAI token limits
if all_chunks:
    num_batches = (len(all_chunks) + batch_size - 1) // batch_size
    print(f"  Processing {len(all_chunks)} chunks in {num_batches} batch(es)...")

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_ids = all_chunk_ids[i:i + batch_size]

        # Add batch to vector store
        vector_store.add_documents(
            documents=batch_chunks,
            ids=batch_ids
        )

        if num_batches > 1:
            batch_num = (i // batch_size) + 1
            print(f"    Batch {batch_num}/{num_batches} complete ({len(batch_chunks)} chunks)")
```

### Step 5: Run Processing

```python
process_html_files(html_dir='italian_sources', force_reprocess=True, batch_size=100)
```

## Expected Output

You should see:

```
+ artart.html - New file, processing...
  Processing 314 chunks in 4 batch(es)...
    Batch 1/4 complete (100 chunks)
    Batch 2/4 complete (100 chunks)
    Batch 3/4 complete (100 chunks)
    Batch 4/4 complete (14 chunks)
  ✓ Title: L'Artusi
    Author: Giovanni Maria Artusi | Date: 1600
    Pages: 155 | Chunks: 314

+ zarins58.html - New file, processing...
  Processing 1027 chunks in 11 batch(es)...
    Batch 1/11 complete (100 chunks)
    Batch 2/11 complete (100 chunks)
    ...
```

## Batch Size Guidelines

- **batch_size=100** (default): Safe for most files, good balance
- **batch_size=50**: Very safe, slower but guaranteed to work
- **batch_size=150**: Faster, but may hit token limits on very long chunks

## Alternative: Quick Copy-Paste Fix

If you want a quick solution, you can also:

1. Open `tmi_fix_complete.py`
2. Copy **CELL 1** into a new cell and run it
3. Copy **CELL 2** into the cell with `process_html_files` and run it
4. Copy **CELL 3** and run it

This will restart the database and process all files with batching.
