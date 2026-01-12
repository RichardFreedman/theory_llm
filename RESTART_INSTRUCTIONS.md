# How to Fix TMI Parser Notebook - Simple Steps

## The Problem
You have a **readonly database error** because the ChromaDB connection is locked by your current Jupyter session.

## The Solution (3 Simple Steps)

### Step 1: Restart the Kernel
In Jupyter, click: **Kernel → Restart Kernel**

This clears the database lock.

### Step 2: Re-run These Cells (in order)
1. Cell 1: Import libraries
2. Cell 2: Enter API key
3. Cell 3: Database configuration
4. Cell 4: Extract metadata and pages functions
5. Cell 5: Helper functions (THIS ONE IS NOW UPDATED WITH BATCH PROCESSING!)

### Step 3: Process Files
Run this:
```python
process_html_files(html_dir='italian_sources', force_reprocess=True)
```

## What's Fixed

✅ **Batch Processing**: The `process_html_files` function now processes chunks in batches of 100
✅ **Token Limit**: Won't exceed OpenAI's 300k token limit anymore
✅ **Large Files**: zarins58.html (1027 chunks) will now process in 11 batches

## Expected Output

```
+ zarins58.html - New file, processing...
  Processing 1027 chunks in 11 batch(es)...
    Batch 1/11 complete (100 chunks)
    Batch 2/11 complete (100 chunks)
    Batch 3/11 complete (100 chunks)
    ...
    Batch 11/11 complete (27 chunks)
  ✓ Title: Le istitutioni harmoniche
    Author: Gioseffo Zarlino | Date: 1558
    Pages: 479 | Chunks: 1027
```

## That's It!

The notebook is now fixed with batch processing built in. Just restart the kernel and re-run the cells.

---

**Note**: I've saved a backup of your original notebook as `TMI_Parser_Chroma_Builder_BACKUP.ipynb` just in case.
