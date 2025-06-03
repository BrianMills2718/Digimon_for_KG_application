#!/usr/bin/env python3
"""
Fix Corpus Path Issue - Convert existing chunks to expected format
"""

import pickle
import json
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append('/home/brian/digimon_cc')

def convert_chunks_to_corpus():
    """Convert existing pickle chunks to JSONL Corpus.json format"""
    print("🔧 Converting chunks to expected corpus format")
    print("=" * 50)
    
    # Load existing chunks from pickle files
    chunk_data_path = Path("results/Fictional_Test/kg_graph/chunk_storage_chunk_data_key.pkl")
    
    if not chunk_data_path.exists():
        print(f"❌ Chunk data not found at: {chunk_data_path}")
        return False
    
    print(f"📁 Loading chunks from: {chunk_data_path}")
    
    try:
        # Load the chunks
        with open(chunk_data_path, 'rb') as f:
            chunks_dict = pickle.load(f)
        
        print(f"✅ Loaded {len(chunks_dict)} chunks")
        
        # Convert to JSONL format
        corpus_documents = []
        for chunk_id, chunk in chunks_dict.items():
            # Extract chunk data
            doc = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "index": chunk.index,
                "tokens": chunk.tokens
            }
            corpus_documents.append(doc)
        
        # Create the output directory
        output_dir = Path("Data/Fictional_Test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSONL format (one JSON object per line)
        corpus_file = output_dir / "Corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for doc in corpus_documents:
                json.dump(doc, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Created corpus file: {corpus_file}")
        print(f"📊 Contains {len(corpus_documents)} documents")
        
        # Verify the file is readable
        with open(corpus_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f if line.strip())
        
        print(f"✅ Verification: {line_count} valid lines in corpus file")
        
        # Show sample
        print(f"\n📝 Sample document:")
        if corpus_documents:
            sample = corpus_documents[0]
            print(f"  Doc ID: {sample['doc_id']}")
            print(f"  Title: {sample['title']}")
            print(f"  Content: {sample['content'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting chunks: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_chunks_to_corpus()
    if success:
        print(f"\n🎉 Corpus conversion successful! CLI should now work.")
    else:
        print(f"\n❌ Corpus conversion failed")